# -*- coding: utf-8 -*-
"""
:mod:`channel.worker` -- Multi-device sync API for a single computation device
==============================================================================

.. module:: worker
   :platform: Unix
   :synopsis: Provide methods for single device Theano code that enable
              homogeneous operations across multiple devices.

Contains :class:`Worker` which provides Platoon's basic API for multi-device
operations. Upon creation, a Worker will initiate connections with its node's
:class:`Controller` process (ZMQ) and get access to intra-node lock. A worker
process is meant to have only one Worker instance to manage a corresponding
computation device, e.g. GPU. Thus, Worker is a singleton class.

Worker's available API depends on available backend frameworks. Currently, there
are two ways to use a Worker for global operations on parameters:

   1. Through :meth:`Worker.sync_params`, which is its default interface.
   2. Or :meth:`Worker.all_reduce` which is a multi-node/GPU collective
      operation.

For detailed information about these methods please check their corresponding
documentation, as well as the brief table which compares the two in project's
:file:`README.md`.

Worker also has :meth:`Worker.recv_mb` interface for collecting mini-batches to
work on from Controller.

"""
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import signal
import base64

import numpy
import posix_ipc
import six
import zmq

try:
    import pygpu
    from pygpu import collectives as gpucoll
    from theano import gpuarray as theanoga
    from theano import config as theanoconf
except ImportError:
    pygpu = None

from ..util import (mmap, PlatoonError, PlatoonWarning, SingletonType)

if six.PY3:
    buffer_ = memoryview
else:
    buffer_ = buffer  # noqa


@six.add_metaclass(SingletonType)
class Worker(object):
    """
    Interface for multi-device operations.

    This class handles communication/synchronization with other processes.
    The features to do so (control channel, mini-batch channel and shared
    parameters) are all independent and optional so you don't have to use all
    of them.

    Parameters
    ----------
    control_port : int
       The tcp port number for control (ZMQ).
    port : int, optional
       The tcp port number for data (ZMQ).
    socket_timeout : int, optional
       Timeout in ms for both control and data sockets. Default: 5 min
    hwm : int, optional
       High water mark (see pyzmq docs) for data transfer.

    Attributes
    ----------
    shared_params : list of :class:`numpy.ndarray`
       This will have `numpy.ndarray` in the same order as `params_descr`
       (see :meth:`init_shared_params`). These arrays are backed by shared
       memory. Used by :meth:`sync_params` interface.
    shared_arrays : dict of str to :class:`numpy.ndarray`
       Maps size in bytes to a ndarray in shared memory. Needed in multi-node
       operations. Used by :meth:`all_reduce` interface.

    """
    def __init__(self, control_port, data_port=None, socket_timeout=300000,
                 data_hwm=10, port=None):
        if port is not None:
            raise RuntimeError(
                "The port parameter of Worker was renamed to data_port"
                " (as in the Controller)")
        self.context = zmq.Context()

        self._socket_timeout = socket_timeout

        self._worker_id = os.getpid()

        if data_port:
            self.init_mb_sock(data_port, data_hwm)

        self._init_control_socket(control_port)

        self._job_uid = self.send_req("platoon-get_job_uid")
        print("JOB UID received from the controler {}".format(self._job_uid))
        self._lock = posix_ipc.Semaphore("{}_lock".format(self._job_uid))

        signal.signal(signal.SIGINT, self._handle_force_close)
        try:
            self._register_to_platoon()
        except Exception as exc:
            print(PlatoonWarning("Failed to register in a local GPU comm world.", exc),
                  file=sys.stderr)
            print(PlatoonWarning("Platoon `all_reduce` interface will not be functional."),
                  file=sys.stderr)
            self._local_comm = None
        self._shmem_names = dict()
        self._shmrefs = dict()
        self.shared_arrays = dict()

################################################################################
#                           Basic Control Interface                            #
################################################################################

    def send_req(self, req, info=None):
        """
        Sends a control request to node's :class:`Controller`.

        Parameters
        ----------
        req : object
           Json-encodable object (usually Python string) that represents the
           type of request being sent to Controller.
        info : object, optional
           Json-encodable object used as input for this Worker's request to
           Controller.

        Returns
        -------
        object
           Json-decoded object.

        """
        query = {'worker_id': self._worker_id, 'req': req, 'req_info': info}
        self.csocket.send_json(query)

        socks = dict(self.cpoller.poll(self._socket_timeout))
        if socks and socks.get(self.csocket) == zmq.POLLIN:
            return self.csocket.recv_json()
        else:
            raise PlatoonError("Control Socket: recv timeout")

    def lock(self, timeout=None):
        """
        Acquire intra-node lock.

        This is advisory and does not prevent concurrent access. This method
        will subtracts 1 in underlying POSIX semaphore, will block the rest
        calls at 0. The underlying semaphore, :attr:`_lock`, starts at 1.

        Parameters
        ----------
        timeout : int, optional
           Amount of time to wait for the lock to be available. A timeout of 0
           will raise an error immediately if the lock is not available.
           Default: None, which will block until the lock is released.

        .. versionchanged:: 0.6.0
           This method used to be called `lock_params`.

        """
        self._lock.acquire(timeout)

    def unlock(self):
        """
        Release intra-node lock.

        The current implementation does not ensure that the process
        that locked :attr:`shared_params` is also the one that unlocks them.
        It also does not prevent one process from unlocking more than once
        (which will allow more than one process to hold the lock). This method
        will add 1 in underlying POSIX semaphore, :attr:`_lock`.

        Make sure you follow proper lock/unlock logic in your program
        to avoid these problems.

        .. versionchanged:: 0.6.0
           This method used to be called `unlock_params`.

        """
        self._lock.release()

    @property
    def local_size(self):
        "Number of workers assigned to local host's controller."
        return self._local_size

    @property
    def local_rank(self):
        "Worker's rank in respect to local host's controller (NCCL comm world)."
        return self._local_rank

    @property
    def global_size(self):
        "Number of workers spawned across all hosts in total."
        return self._global_size

    @property
    def global_rank(self):
        "Worker's rank in respect to all hosts' controllers in total."
        return self._global_rank

################################################################################
#                   Initialization and Finalization Methods                    #
################################################################################

    def _handle_force_close(self, signum, frame):
        """Handle SIGINT signals from Controller.

        This is expected to happen when something abnormal has happened in other
        workers which implies that training procedure should stop and fail.

        .. versionadded:: 0.6.0

        """
        self.close()
        sys.exit(1)  # Exit normally with non success value.

    def close(self):
        """
        Closes ZMQ connections, POSIX semaphores and shared memory.
        """
        print("Closing connections and unlinking memory...", file=sys.stderr)
        if hasattr(self, 'asocket'):
            self.asocket.close()
        if hasattr(self, 'csocket'):
            self.csocket.close()
        self.context.term()
        self._lock.close()
        try:
            self._lock.unlink()
        except posix_ipc.ExistentialError:
            pass
        if hasattr(self, '_shmref'):
            try:
                self._shmref.unlink()
            except posix_ipc.ExistentialError:
                pass
        for shmref in self._shmrefs.values():
            try:
                shmref.unlink()
            except posix_ipc.ExistentialError:
                pass

    def _register_to_platoon(self):
        """
        Asks Controller for configuration information and creates a NCCL
        communicator that participate in the local node's workers world.

        For this it is needed that Theano is imported. Through Theano, this
        methods gets access to the single GPU context of this worker process.
        This context is to be used in all computations done by a worker's
        process.

        .. note::
           It is necessary that this initialization method is called
           successfully before :meth:`all_reduce` in order to be available
           and functional.

        .. versionadded:: 0.6.0

        """
        if pygpu:
            self.ctx_name = None
            self.gpuctx = theanoga.get_context(self.ctx_name)
            self.device = theanoconf.device
            self._local_id = gpucoll.GpuCommCliqueId(context=self.gpuctx)
            # Ask controller for local's info to participate in
            lid = base64.b64encode(self._local_id.comm_id).decode('ascii')
            response = self.send_req("platoon-get_platoon_info",
                                     info={'device': self.device,
                                           'local_id': lid})
            nlid = base64.b64decode(response['local_id'].encode('ascii'))
            self._local_id.comm_id = bytearray(nlid)
            self._local_size = response['local_size']
            self._local_rank = response['local_rank']
            self._local_comm = gpucoll.GpuComm(self._local_id,
                                               self._local_size,
                                               self._local_rank)
            self._multinode = response['multinode']
            self._global_size = response['global_size']
            self._global_rank = response['global_rank']
        else:
            raise AttributeError("pygpu or theano is not imported")

    def init_mb_sock(self, port, data_hwm=10):
        """
        Initialize the mini-batch data socket.

        Parameters
        ----------
        port : int
           The tcp port to reach the mini-batch server on.
        data_hwm : int, optional
           High water mark, see pyzmq docs.

        .. note::
           This must be called before using :meth:`recv_mb`.

        """
        self.asocket = self.context.socket(zmq.PULL)
        self.asocket.setsockopt(zmq.LINGER, 0)
        self.asocket.set_hwm(data_hwm)
        self.asocket.connect("tcp://localhost:{}".format(port))

        self.apoller = zmq.Poller()
        self.apoller.register(self.asocket, zmq.POLLIN)

    def _init_control_socket(self, port):
        """
        Initialize control socket.

        Parameters
        ---------
        port : int
          The tcp port where the control master is listening at.

        .. note::
           This must be called before using :meth:`send_req`.

        """
        self.csocket = self.context.socket(zmq.REQ)
        self.csocket.setsockopt(zmq.LINGER, 0)
        self.csocket.connect('tcp://localhost:{}'.format(port))

        self.cpoller = zmq.Poller()
        self.cpoller.register(self.csocket, zmq.POLLIN)

################################################################################
#                            Collectives Interface                             #
################################################################################

    def shared(self, array):
        """Creates a new POSIX shared memory buffer to be shared among Workers
        and their Controller and maps the size of `array` to that buffer.

        Controller is requested to create a new shared memory buffer with the
        same size as `array` in order to be used in multi-GPU/node Platoon
        collective operations through :meth:`all_reduce` interface. All
        participants in the same node have access to that memory.

        :param array: This array's size in bytes will be mapped to a shared
                      memory buffer in host with the same size.
        :type array: :ref:`pygpu.gpuarray.GpuArray`

        Returns
        -------
        shared_array : :ref:`numpy.ndarray`
           A newly created shared memory buffer with the same size or an already
           allocated one.

        Notes
        -----
        *For internal implementation*: There should probably be a barrier across
        nodes' Workers to ensure that, so far, each Controller has serviced
        a new shared memory's name to all Workers. This is due to the fact that
        Controller can service one Worker at a time and a Platoon collective
        service is a blocking one across Controllers. Current implementation
        is valid because calls to `pygpu.collectives` interface are synchronous
        across workers.

        .. versionadded:: 0.6.0

        """
        if not isinstance(array, pygpu.gpuarray.GpuArray):
            raise TypeError("`array` input is not pygpu.gpuarray.GpuArray.")

        # This is not a problem, unless we have concurrent calls in
        # :meth:`all_reduce` in the same worker-process and we are running in
        # multi-node. This due to the fact that :attr:`shared_arrays` are being
        # used as temporary buffers for the internal inter-node MPI collective
        # operation. We only need a shared buffer with Controller in order to
        # execute multi-node operation, so a mapping with size in bytes
        # suffices. See:
        # https://github.com/mila-udem/platoon/pull/66#discussion_r74988680
        bytesize = array.size * array.itemsize

        if bytesize in self.shared_arrays:
            return self.shared_arrays[bytesize]
        else:
            if array.flags['F']:
                order = 'F'
            else:
                order = 'C'

            try:
                shared_mem_name = self.send_req("platoon-init_new_shmem",
                                                info={'size': bytesize})

                shmref = posix_ipc.SharedMemory(shared_mem_name)
                shm = mmap(fd=shmref.fd, length=bytesize)
                shmref.close_fd()
            except Exception as exc:
                try:
                    shmref.unlink()
                except (NameError, posix_ipc.ExistentialError):
                    pass
                raise PlatoonError("Failed to get access to shared memory buffer.", exc)
            shared_array = numpy.ndarray(array.shape, dtype=array.dtype,
                                         buffer=shm, offset=0, order=order)
            self._shmem_names[bytesize] = shared_mem_name  # Keep for common ref with Controller
            self._shmrefs[bytesize] = shmref  # Keep for unlinking when closing
            self.shared_arrays[bytesize] = shared_array
            return shared_array

    def all_reduce(self, src, op, dest=None):
        """
        AllReduce collective operation for workers in a multi-node/GPU Platoon.

        Parameters
        ----------
        src : :ref:`pygpu.gpuarray.GpuArray`
           Array to be reduced.
        op : str
           Reference name to reduce operation type.
           See :ref:`pygpu.collectives.TO_RED_OP`.
        dest : :ref:`pygpu.gpuarray.GpuArray`, optional
           Array to collect reduce operation result.

        Returns
        -------
        result: None or :ref:`pygpu.gpuarray.GpuArray`
           New Theano gpu shared variable which contains operation result
           if `dest` is None, else nothing.

        .. warning::
           Repeated unnecessary calls with no `dest`, where a logically valid
           pygpu GpuArray exists, should be avoided for optimal performance.

        .. versionadded:: 0.6.0

        """
        if self._local_comm is None:
            raise PlatoonError("`all_reduce` interface is not available. Check log.")

        if not isinstance(src, pygpu.gpuarray.GpuArray):
            raise TypeError("`src` input is not pygpu.gpuarray.GpuArray.")

        if dest is not None:
            if not isinstance(dest, pygpu.gpuarray.GpuArray):
                raise TypeError("`dest` input is not pygpu.gpuarray.GpuArray.")

        try:
            # Execute collective operation in local NCCL communicator world
            res = self._local_comm.all_reduce(src, op, dest)
        except Exception as exc:
            raise PlatoonError("Failed to execute pygpu all_reduce", exc)

        if dest is not None:
            res = dest
        res.sync()

        # If running with multi-node mode
        if self._multinode:
            # Create new shared buffer which corresponds to result GpuArray buffer
            res_array = self.shared(res)

            self.lock()
            first = self.send_req("platoon-am_i_first")
            if first:
                # Copy from GpuArray to shared memory buffer
                res.read(res_array)
                res.sync()

                # Request from controller to perform the same collective operation
                # in MPI communicator world using shared memory buffer
                self.send_req("platoon-all_reduce", info={'shmem': self._shmem_names[res.size * res.itemsize],
                                                          'dtype': str(res.dtype),
                                                          'op': op})
            self.unlock()

            # Concurrently copy from shared memory back to result GpuArray
            # after Controller has finished global collective operation
            res.write(res_array)
            res.sync()

        if dest is None:
            return res

################################################################################
#                             Param Sync Interface                             #
################################################################################

    def _get_descr_size(self, dtype, shape):
        size = dtype.itemsize
        for s in shape:
            size *= s
        return size

    def init_shared_params(self, params, param_sync_rule):
        """
        Initialize shared memory parameters.

        This must be called before accessing the params attribute
        and/or calling :meth:`sync_params`.

        Parameters
        ----------
        params : list of :ref:`theano.compile.SharedVariable`
           Theano shared variables representing the weights of your model.
        param_sync_rule : :class:`param_sync.ParamSyncRule`
           Update rule for the parameters

        """
        self.update_fn = param_sync_rule.make_update_function(params)
        self.local_params = params

        params_descr = [(numpy.dtype(p.dtype), p.get_value(borrow=True).shape)
                        for p in params]
        params_size = sum(self._get_descr_size(*d) for d in params_descr)

        shared_mem_name = "{}_params".format(self._job_uid)

        # Acquire lock to decide who will init the shared memory
        self.lock()

        need_init = self.send_req("platoon-need_init")
        if need_init:
            # The ExistentialError is apparently the only way to verify
            # if the shared_memory exists.
            try:
                posix_ipc.unlink_shared_memory(shared_mem_name)
            except posix_ipc.ExistentialError:
                pass

            self._shmref = posix_ipc.SharedMemory(shared_mem_name,
                                                  posix_ipc.O_CREAT,
                                                  size=params_size)
        else:
            self._shmref = posix_ipc.SharedMemory(shared_mem_name)

        self._shm = mmap(fd=self._shmref.fd, length=params_size)
        self._shmref.close_fd()
        self.shared_params = []
        off = 0

        for dtype, shape in params_descr:
            self.shared_params.append(numpy.ndarray(shape, dtype=dtype,
                                                    buffer=self._shm,
                                                    offset=off))
            off += self._get_descr_size(dtype, shape)

        if need_init:
            self.copy_to_global(synchronous=False)

        self.unlock()

    def sync_params(self, synchronous=True):
        """
        Update the worker's parameters and the central parameters according
        to the provided parameter update rule.

        Parameters
        ----------
        synchronous : bool
           If false, the lock won't be acquired before touching the
           shared weights.

        """
        if synchronous:
            self.lock()

        self.update_fn(self.shared_params)

        if synchronous:
            self.unlock()

    def copy_to_local(self, synchronous=True):
        """
        Copy the global params to the local ones.

        Parameters
        ----------
        synchronous : bool
           If False, the lock won't be acquired before touching the
           shared weights.

        """
        if synchronous:
            self.lock()

        for p, v in zip(self.local_params, self.shared_params):
            p.set_value(v)

        if synchronous:
            self.unlock()

    def copy_to_global(self, synchronous=True):
        """
        Copy the global params to the local ones.

        Parameters
        ----------
        synchronous : bool
           If False, the lock won't be acquired before touching the
           shared weights.

        """
        if synchronous:
            self.lock()

        for p, v in zip(self.local_params, self.shared_params):
            v[:] = p.get_value(borrow=True)

        if synchronous:
            self.unlock()

################################################################################
#                           Distribute Data Batches                            #
################################################################################

    def recv_mb(self):
        """
        Receive a mini-batch for processing.

        A mini-batch is composed of a number of numpy arrays.

        Returns
        -------
        list
           The list of numpy arrays for the mini-batch

        """
        socks = dict(self.apoller.poll(self._socket_timeout))
        if socks:
            if socks.get(self.asocket) == zmq.POLLIN:
                headers = self.asocket.recv_json()
        else:
            raise Exception("Batch socket: recv timeout")

        arrays = []
        for header in headers:

            data = self.asocket.recv(copy=False)

            buf = buffer_(data)
            array = numpy.ndarray(
                buffer=buf, shape=header['shape'],
                dtype=numpy.dtype(header['descr']),
                order='F' if header['fortran_order'] else 'C')
            arrays.append(array)
        return arrays

    @staticmethod
    def default_parser():
        """
        Returns base :class:`Controller`'s class parser for its arguments.

        This parser can be augmented with more arguments, if it is needed, in
        case a class which inherits :class:`Controller` exists.

        .. versionadded:: 0.6.1

        """
        parser = argparse.ArgumentParser(
            description="Base Platoon Worker process.")
        parser.add_argument('--control-port', default=5567, type=int, required=False, help='The control port number.')
        parser.add_argument('--data-port', type=int, required=False, help='The data port number.')
        parser.add_argument('--data-hwm', default=10, type=int, required=False, help='The data port high water mark')
        return parser

    @staticmethod
    def default_arguments(args):
        """
        Static method which returns the correct arguments for a base
        :class:`Controller` class.

        :param args:
           Object returned by calling :meth:`argparse.ArgumentParser.parse_args`
           to a parser returned by :func:`default_parser`.

        .. versionadded:: 0.6.0

        """
        DEFAULT_KEYS = ['control_port', 'data_hwm', 'data_port']
        d = args.__dict__
        return dict((k, d[k]) for k in six.iterkeys(d) if k in DEFAULT_KEYS)
