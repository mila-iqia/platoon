import os

import numpy
import posix_ipc
import six
import zmq
from pygpu import collectives as gpucoll

from mmap import mmap

# You need:
# $ conda install pyzmq cffi
# $ pip install posix_ipc

# Also this was only tested in python 2.7

if six.PY3:
    buffer_ = memoryview
else:
    buffer_ = buffer  # noqa


class Worker(object):
    """
    Worker object. Each worker should have one instance of this class.

    This class handles the communication/synchronization with other processes.
    The features to do so (control channel, mini-batch channel and shared
    parameters) are all independent and optional so you don't have to use all
    of them.

    Parameters
    ----------
    port : int, optional
        Will call :meth:`init_mb_sock` with this port.
    control_port : int
        Will call :meth:`_init_control_socket` with this port.
    socket_timeout : int
        Timeout in ms for both sockets. Default: 5 min
    hwm : int
        High water mark (see pyzmq docs).

    Attributes
    ----------
    params : list of ndarrays
        This will have numpy ndarray in the same order as params_descr.
        These arrays are backed by shared memory.

    """

    def __init__(self, control_port, local_rank, port=None, socket_timeout=300000, hwm=10):
        self.context = zmq.Context()

        self._socket_timeout = socket_timeout

        self._worker_id = os.getpid()
        self._local_rank = local_rank

        if port:
            self.init_mb_sock(port, hwm)

        self._init_control_socket(control_port)

        self._job_uid = self.send_req("platoon-get_job_uid")
        self._device_name = self.send_req("platoon-get_device",
                                          info={'local_rank': self._local_rank})
        self._lock = posix_ipc.Semaphore("{}lock".format(self._job_uid))

        self._register_to_platoon()
        self._shmrefs = dict()
        self.shared_arrays = dict()

    def send_req(self, req, info=None):
        """
        Send a control request.

        Parameters
        ----------
        req : object
            This is a json-encodable object that will be sent to the
            Controller.
        info : object

        Returns
        -------
        object
            Json-decoded object

        """
        query = {'worker_id': self._worker_id, 'req': req, 'req_info': info}
        self.csocket.send_json(query)

        socks = dict(self.cpoller.poll(self._socket_timeout))
        if socks and socks.get(self.csocket) == zmq.POLLIN:
            return self.csocket.recv_json()
        else:
            raise Exception("Control Socket: recv timeout")

    def close(self):
        if hasattr(self, 'asocket'):
            self.asocket.close()
        if hasattr(self, 'csocket'):
            self.csocket.close()
        if hasattr(self, '_shmref'):
            self._lock.close()
            try:
                self._lock.unlink()
            except posix_ipc.ExistentialError:
                pass
            try:
                self._shmref.unlink()
            except posix_ipc.ExistentialError:
                pass

    def _register_to_platoon(self):
        # TODO Set Theano device to be used by Theano
        # TODO Get pygpu context used by Theano
        gpuctx = None
        self._region_id = gpucoll.GpuCommCliqueId(context=gpuctx)
        # Ask controller for region's info to participate in
        response = self.send_req("platoon-get_region_info",
                                 info={'local_rank': self._local_rank,
                                       'region_id': self._region_id.comm_id})
        self._region_id.comm_id = bytearray(response['region_id'])
        self._regional_comm = gpucoll.GpuComm(self._region_id,
                                              response['region_size'],
                                              response['regional_rank'])

    def init_mb_sock(self, port, hwm=10):
        """
        Initialize the mini-batch socket.

        This must be called before using :meth:`recv_mb`.

        Parameters
        ----------
        port : int
            The port to reach the mini-batch server on.
        hwm : int
            High water mark, see pyzmq docs.

        """
        self.asocket = self.context.socket(zmq.PULL)
        self.asocket.setsockopt(zmq.LINGER, 0)
        self.asocket.set_hwm(hwm)
        self.asocket.connect("tcp://localhost:{}".format(port))

        self.apoller = zmq.Poller()
        self.apoller.register(self.asocket, zmq.POLLIN)

    def _init_control_socket(self, port):
        """
        Initialize control socket.

        This must be called before using :meth:`send_req`.

        Parameters
        ---------
        port : int
            Port where the control master is listening on.

        """
        self.csocket = self.context.socket(zmq.REQ)
        self.csocket.setsockopt(zmq.LINGER, 0)
        self.csocket.connect('tcp://localhost:{}'.format(port))

        self.cpoller = zmq.Poller()
        self.cpoller.register(self.csocket, zmq.POLLIN)

    def lock(self, timeout=None):
        """
        Acquire the lock across all workers.

        This is advisory and does not prevent concurrent access.
        Subtracts 1 in underlying semaphore. Blocks at 0. Begins at 1.

        Parameters
        ----------
        timeout : int
            Amount of time to wait for the lock to be available.  A
            timeout of 0 will raise an error immediately if the lock is
            not available.

        """
        self._lock.acquire(timeout)

    def unlock(self):
        """
        Release the lock across all workers.

        The current implementation does not ensure that the process
        that locked the params is the one that unlocks them. It also
        does not prevent one process from unlocking more than once
        (which will allow more than one process to hold the lock). Adds 1 in
        underlying semaphore.

        Make sure you follow proper lock/unlock logic in your program
        to avoid these problems.
        """
        self._lock.release()


################################################################################
#                            New Control Interface                             #
################################################################################

    def new_linked_shared(self, gpuarray):
        # TODO Get a hashable Theano variable and fetch pygpu internal
        size = gpuarray.size * gpuarray.itemsize
        shared_mem_name = self.send_req("platoon-init_new_shmem",
                                        info={'size': size})
        #  self.lock()

        shmref = posix_ipc.SharedMemory(shared_mem_name)
        shm = mmap(fd=shmref.fd, length=size)
        shmref.close_fd()
        array = numpy.ndarray(gpuarray.shape, dtype=gpuarray.dtype,
                              buffer=shm, offset=0)  # give order?
        self._shmem_names[gpuarray] = shared_mem_name
        self._shmrefs[gpuarray] = shmref  # Keep for unlinking when closing
        self.shared_arrays[gpuarray] = array

        #  self.unlock()

    def all_reduce(self, src, op, dest=None):
        # TODO Get a hashable Theano variable and fetch pygpu internal
        # perform collective among NCCL gpu region using pygpu
        res = self._regional_comm.all_reduce(src, op, dest)

        # if multi-node is needed then do the following (define better
        # condition):
        if dest is None:
            self.new_linked_shared(res)
        else:
            if dest not in self.shared_arrays:
                self.new_linked_shared(dest)
            res = dest

        res_array = self.shared_arrays[res]

        self.lock()
        first = self.send_req("platoon-am_i_first")
        if first:
            # write from gpuarray to shared memory
            res.read(res_array)
            # ask controller to perform the same collective on shared memory
            # using MPI
            self.send_req("platoon-all_reduce", info={'shmem': self._shmem_names[res],
                                                      'dtype': str(res.dtype),
                                                      'op': op})
        self.unlock()

        # simultaneously read from shared memory back to result gpuarray
        # only after region Controller has finished global collective operation
        res.write(res_array)

        if dest is None:
            return res

################################################################################
#                      Old Control Interface (param sync)                      #
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
        and/or calling :meth:`sync_params`, :meth:`lock` or
        :meth:`unlock`.

        Parameters
        ----------
        params : shared variables
            Theano shared variables representing the weights of your model.
        param_sync_rule : ParamSyncRule
            Update rule for the parameters
        cleanup : bool
            Whether to cleanup a previous run with the same
            identifier.  Will also copy the current values of `params`
            to the shared memory.  This is required on certain
            platform due to system restrictions.

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
