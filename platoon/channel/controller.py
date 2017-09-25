# -*- coding: utf-8 -*-
"""
:mod:`channel.controller` -- Coordination of multiple Platoon Worker processes
==============================================================================

.. module:: controller
   :platform: Unix
   :synopsis: Control Workers and execute their requests in a multi-node/process
              system.

Contains :class:`Controller` and helper functions to spawn an instance of this
class. Upon creation, a Controller will initiate connections (ZMQ and possibly
MPI), setup intra-node lock and spawn worker processes as its children,
according to the given arguments and/or configuration.

This file can also be executed, spawning a Controller instance and making
serve until all of its children processes exit or an error occurs. It will be
executed in a separate process as a default by :file:`script/platoon-launcher`,
if user has not provided another executable (by the name
'<experiment>_controller.py') which will spawn base :class:`Controller` or a
decedent class.

"""
from __future__ import absolute_import, print_function
import os
import sys
import signal
import time
import shlex

import six
from six.moves import range

import argparse
import numpy
import posix_ipc
import zmq

try:
    from mpi4py import rc
    rc.initialize = False
    from mpi4py import MPI
except ImportError:
    MPI = None

from ..util import (PlatoonError, mmap, launch_process,
                    op_to_mpi, dtype_to_mpi)


class Controller(object):
    """
    General multi-process controller.

    This class provides the necessary features to dispatch data mini-batches
    to workers and handle control requests.

    .. warning::
       Due to the underlying implementation it is a bad idea to
       attempt to do both in the same process, even on different
       threads. This will suffer from interlock problems and may
       negate any speedup you could get from using multiple :class:`Worker`s.

       Because of this issue, the class may be split in the future.

    Parameters
    ----------
    control_port : int, optional
       The tcp port number for control (ZMQ).
    data_port : int, optional
       The tcp port number for data (ZMQ).
    data_hwm : int, optional
       High water mark (see pyzmq docs) for data transfer.
    devices : list of strings, optional
       Contains device names in clique order (prefer ring topology).
    experiment_name : str, optional
       Name of experiment given in :file:`platoon_launcher`. Will find
       `experiment_name`_worker.py with this.
    log_directory : str, optional
       Directory in which this controller's workers will create log files.
    worker_args : str, optional
       Arguments for workers specified in :file:`platoon_launcher`, if any.
    multi : bool, optional
       True, if we start a multi-node experiment. Flag to start MPI.

    """
    def __init__(self, control_port=5567, data_port=None, data_hwm=10,
                 devices=None, experiment_name='',
                 log_directory='', worker_args='', multi=False):
        self._workers = set()
        self._need_init = True

        self._devices = list()
        if devices is not None:
            self._devices = Controller.get_workers_devices(devices)
        self._local_size = len(self._devices)
        self._global_size = self._local_size
        self._get_platoon_info_count = [0]
        self._init_new_shmem_count = [0]
        self._am_i_first_count = [0]

        signal.signal(signal.SIGTERM, self._handle_force_close)
        signal.signal(signal.SIGINT, self._handle_force_close)

        # If we are starting a multi-node training
        self._multinode = multi
        if self._multinode:
            print("## Running in multi-node mode.")
            try:
                self._init_region_comm()
            except Exception as exc:
                print("WARNING! {} while being in multi-node mode".format(exc),
                      file=sys.stderr)
                self._region_comm = None

        if data_port:
            self.init_data(data_port, data_hwm)

        self._init_control_socket(control_port)

        # Cleanup and init local lock and job_uid name
        self._job_uid = "platoon_{0}_{1}".format(
            os.path.basename(os.path.expanduser('~')), control_port)

        self._lock_name = "{}_lock".format(self._job_uid)
        # The ExistentialError is apparently the only way to verify if
        # the semaphore/shared_memory exists.
        try:
            posix_ipc.unlink_semaphore(self._lock_name)
        except posix_ipc.ExistentialError:
            pass
        # Initializing lock
        self._lock = posix_ipc.Semaphore(self._lock_name, posix_ipc.O_CREAT,
                                         initial_value=1)

        # Init dictionaries for shared buffers with workers
        self._shmrefs = dict()
        self.shared_buffers = dict()

        # Initialize workers
        if experiment_name:
            try:
                os.makedirs(log_directory)
            except OSError:
                pass
            if worker_args is None:
                worker_args = ''
            worker_args += " --control-port=%d" % control_port
            worker_args += " --data-hwm=%d" % data_hwm
            if data_port:
                worker_args += " --data-port=%d" % data_port
            try:
                for device in self._devices:
                    p = launch_process(log_directory, experiment_name,
                                       shlex.split(worker_args), device,
                                       "worker")
                    self._workers.add(p.pid)
            except OSError as exc:
                print("ERROR! OS error in Popen: {}".format(exc), file=sys.stderr)
                sys.exit(3)
            except Exception as exc:
                print("ERROR! Other while launching process: {}".format(exc), file=sys.stderr)
                sys.exit(4)

################################################################################
#                         Control Serving and Handling                         #
################################################################################

    def handle_control(self, req, worker_id, req_info):
        """
        Re-implement or assign a handler to this function to do
        something with control messages.

        The replacement get one parameter which is the request and
        should return the response which must be a json-encodable
        object. Other code is responsible for handling decoding,
        encoding and the network.

        """
        raise NotImplementedError("Request type '{0}' is not developed in class "
                                  "{1}. A class should inherit {1} and "
                                  "override method `handle_control` in order to add "
                                  "behavior.".format(req, self.__class__.__name__))

    def _handle_base_control(self, req, worker_id, req_info):
        """
        This method handles base control commands.

        In most cases it acts as a switcher to :class:`Controller`'s methods
        which execute the command `req`.

        Parameters
        ----------
        req : str
           Request type.
        worker_id : int
           Requesting :class:`Worker`'s process id.
        req_info : dict
           Contains :class:`Worker`'s input for its request to :class:`Controller`.

        .. note::
           Those commands should not be used in the handle_control method.
           All base control commands should start with "platoon-".

        """
        response = None
        if req == "platoon-get_job_uid":
            response = self._job_uid

        elif req == "platoon-need_init":
            response = self._need_init
            self._need_init = False

        elif req == "platoon-am_i_first":
            response = self._is_worker_first(self._am_i_first_count)

        elif req == "platoon-get_platoon_info":
            response = self._get_platoon_info(req_info)

        elif req == "platoon-init_new_shmem":
            response = self._init_new_shmem(req_info)

        elif req == "platoon-all_reduce":
            response = self._all_reduce(req_info)

        return response

    def worker_is_done(self, worker_id):
        """
        This method is not supported anymore.

        Rationale
        ---------
        1. For supporting multi-node controllers it was necessary to move the
           spawning of worker process to :class:`Controller` object, so
           :class:`Controller` needs to wait for the exit of its worker
           children. This was done for the following reasons:

           * Controller processes may be MPI processes, while worker processes
              are not.
           * In multi-node case, worker processes have to be spawned in a
             another host than the one executing the launcher.
           * It would be better, if there was a uniform way to spawn worker
             process which is independent whether we are in the single-node or
             multi-node case.

        2. Given (1): Each process ought to take care cleaning and exiting
           gracefully by itself, unless a fatal error has happened or a
           irrecoverable error for the procedure has happened, in which case
           process should exit normally with non-success code. In fatal or
           irrecoverable cases, their parent process (i.e controller) must kill
           them (forcing them to exit gracefully, if possible).

        .. deprecated:: 0.6.0
           A worker process should exit normally instead. It is not needed
           to signal to its controller that it has finished.
        """
        self._should_stop = True

    def serve(self):
        """This method will handle control messages until an error happens or
        all children-worker processes exit.

        Handles controller finilization  by cleaning, logging and returning
        if controller was successful or not.

        """
        try:  # spin spin spin
            self._success = 2
            while self._workers:  # spin while we have still children to watch for
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except OSError as exc:
                    raise PlatoonError("while waiting for a child", exc)
                if pid != 0:  # If a status change has happened at a child
                    if os.WIFEXITED(status):
                        self._workers.discard(pid)
                        self._success = os.WEXITSTATUS(status)
                        if self._success == 0:
                            # A worker has terminated normally. Other workers
                            # are expected to terminate normally too, so
                            # continue.
                            continue
                        else:
                            # A worker has not terminated normally due to an
                            # error or an irrecoverable fault
                            raise PlatoonError("A worker has exited with non-success code: {}".format(self._success))
                    else:  # other status changes are not desirable
                        raise PlatoonError("A worker has changed to a status other than exit.")
                try:
                    query = self.csocket.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:  # if a query has not happened, try again
                    continue
                except zmq.ZMQError as exc:
                    raise PlatoonError("while receiving using ZMQ socket", exc)

                # try default interface, it may raise PlatoonError
                response = self._handle_base_control(query['req'],
                                                     query['worker_id'],
                                                     query['req_info'])
                if response is None:
                    response = self.handle_control(query['req'],
                                                   query['worker_id'],
                                                   query['req_info'])

                try:
                    self.csocket.send_json(response)
                except zmq.ZMQError as exc:
                    raise PlatoonError("while sending using ZMQ socket", exc)
        except PlatoonError as exc:  # if platoon fails kill all children workers
            print(exc, file=sys.stderr)
            self._clean()
        except Exception as exc:
            print(PlatoonError("Unexpected exception", exc), file=sys.stderr)
            self._clean()
        else:
            if self._multinode and MPI:
                MPI.Finalize()
        finally:  # Close sockets and unlink for shared memory
            self._close()
        return self._success

################################################################################
#                   Initialization and Finalization Methods                    #
################################################################################

    def _init_control_socket(self, port):
        """
        Initialize the control socket.

        This must be called before using :meth:`serve`.

        Parameters
        ----------
        port : int
           The port to listen on.

        """
        self.ccontext = zmq.Context()
        self.csocket = self.ccontext.socket(zmq.REP)
        self.csocket.bind('tcp://*:{}'.format(port))

    def _init_region_comm(self):
        """
        If in multi-node, this method will initialize information about MPI
        controllers.

        .. versionadded:: 0.6.0

        """
        if MPI is None:
            raise AttributeError("mpi4py is not imported")
        MPI.Init()
        self._region_comm = MPI.COMM_WORLD
        self._region_size = MPI.COMM_WORLD.Get_size()
        self._region_rank = MPI.COMM_WORLD.Get_rank()

        local_size = numpy.array([self._local_size], dtype='int32')
        self._all_local_size = numpy.zeros((self._region_size, ), dtype='int32')
        self._region_comm.Allgather([local_size, MPI.UNSIGNED_INT],
                                    [self._all_local_size, MPI.UNSIGNED_INT])
        self._global_size = sum(self._all_local_size)

    def _clean(self):
        """
        Helper method used to kill remaining workers and abort MPI job
        (if we are running in multi-node), in case of error.

        .. versionadded:: 0.6.0

        """
        print("Cleaning up...", file=sys.stderr)
        self._kill_workers()
        if self._multinode and self._region_comm:
            print("Aborting MPI job...", file=sys.stderr)
            self._region_comm.Abort(errorcode=1)

    def _kill_workers(self):
        """
        Kills own remaining children, worker processes.

        .. versionadded:: 0.6.0

        """
        while self._workers:
            pid = self._workers.pop()
            print("Killing worker {}...".format(pid), file=sys.stderr)
            for _ in range(3):
                try:
                    os.kill(pid, signal.SIGINT)
                    break
                except OSError:
                    pass
            try:
                pid, status = os.waitpid(pid, 0)
            except OSError:
                pass

    def _close(self):
        """
        Closes ZMQ connections, POSIX semaphores and shared memory.
        """
        print("Closing connections and unlinking memory...", file=sys.stderr)
        self.csocket.close()
        self.ccontext.term()
        if hasattr(self, 'asocket'):
            self.asocket.close()
            self.acontext.term()
        try:
            self._lock.close()
            self._lock.unlink()
        except posix_ipc.ExistentialError:
            pass
        for shmref in self._shmrefs.values():
            try:
                shmref.unlink()
            except posix_ipc.ExistentialError:
                pass

    def _handle_force_close(self, signum, frame):
        """Handle SIGTERM and SIGINT signals from MPI.Abort.

        This is expected to happen when something abnormal has happened in other
        controllers over MPI.COMM_WORLD across host which participate in the
        multi-node training.

        .. versionadded:: 0.6.0

        """
        print("Caught signal {}. Killing workers and closing connections...".format(
              signum), file=sys.stderr)
        self._kill_workers()
        self._close()
        sys.exit(1)

################################################################################
#                               Control Requests                               #
################################################################################

    def _is_worker_first(self, counter):
        """
        Returns True, if in a mass request in a local platoon (workers in a
        single host) a worker's request reaches first its controller.

        This will work only if every single worker participates successfully
        each time in a concurrent request of the same type to their controller.

        .. versionadded:: 0.6.0

        """
        if self._local_size == 1:
            return True
        counter[0] = (counter[0] + 1) % self._local_size
        if counter[0] == 1:
            return True
        return False

    def _get_platoon_info(self, req_info):
        """
        Packs information about current Platoon's setup.

        Parameters
        ----------
        req_info : dict
           request info from individual :class:`Worker`.

        Returns
        -------
        response : dict
           response to :class:`Worker`'s request.

        req_info
        --------
        * *local_id* : str
           Id string meant to be unique among :class:`Worker`s in the same host.
           This one is currently different for each :class:`Worker`. Produced
           by a NCCL helper function.
        * *device* : str
           Theano device identifier, e.g. cuda2.

        response
        --------
        * *local_id* : str
           Id string unique among :class:`Worker`s in the same host. It is used
           to initialize local NCCL communicator world.
        * *local_size* : int
           Number of :class:`Worker`s spawned in requesting :class:`Worker`'s
           host by :class:`Controller`.
        * *local_rank* : int
           Rank of requesting :class:`Worker` in respect to the local NCCL
           comm world.
        * *multinode* : bool
           True, if we are running a multi-node procedure.
        * *global_size* : int
           Number of :class:`Worker`s spawned across all hosts by
           all :class:`Controller`s in total.
        * *global_rank* : int
           Rank of requesting :class:`Worker` in respect to :class:`Controller`s'
           MPI comm world and their :class:`Worker`s' local NCCL comm world.

        .. versionadded:: 0.6.0

        """
        first = self._is_worker_first(self._get_platoon_info_count)  # See :meth:`_is_worker_first`
        if first:
            self._local_id = req_info['local_id']
        response = dict()
        response['local_id'] = self._local_id
        response['local_size'] = self._local_size
        local_rank = self._devices.index(req_info['device'])
        response['local_rank'] = local_rank
        response['multinode'] = self._multinode
        response['global_size'] = self._global_size
        if self._multinode:
            response['global_rank'] = sum(self._all_local_size[:self._region_rank]) + local_rank
        else:
            response['global_rank'] = local_rank
        return response

    def _init_new_shmem(self, req_info):
        """
        Initiates a POSIX shared memory buffer on a :class:`Worker`'s request
        which will be accesible to all :class:`Worker`s in a host.

        Parameters
        ----------
        req_info : dict
           Request info from individual :class:`Worker`.

        Returns
        -------
        response : str
           Response to a :class:`Worker`'s request. Contains common reference
           name to POSIX shared memory.

        req_info
        --------
        * *size* : int
           Shared memory's size in bytes.

        .. versionadded:: 0.6.0

        """
        first = self._is_worker_first(self._init_new_shmem_count)  # See :meth:`_is_worker_first`
        if first:
            self._last_shmem_name = "platoon-{0}_{1}_buffer".format(self._job_uid,
                                                                    len(self.shared_buffers))
            try:
                posix_ipc.unlink_shared_memory(self._last_shmem_name)
            except posix_ipc.ExistentialError:
                pass

            size = req_info['size']
            try:
                shmref = posix_ipc.SharedMemory(self._last_shmem_name,
                                                posix_ipc.O_CREAT,
                                                size=size)
                shm = mmap(fd=shmref.fd, length=size)
                shmref.close_fd()
            except Exception as exc:
                try:
                    shmref.unlink()
                except (NameError, posix_ipc.ExistentialError):
                    pass
                raise PlatoonError("Failed to initialize new shared buffer.", exc)
            # We want every worker to get the same shared memory name that is
            # was declared in the first call of a mass request to this
            # controller for initializing a new shared memory.
            self._shmrefs[self._last_shmem_name] = shmref
            # Keep for unlinking when closing
            self.shared_buffers[self._last_shmem_name] = shm
        return self._last_shmem_name

    def _all_reduce(self, req_info):
        """
        Request for an AllReduce collective operation on a shared memory buffer
        among every nodes' :class:`Controller`s.

        For this operation to be successful it is needed that a :class:`Worker`
        has written to the shared memory buffer referenced by `shmem`.
        The result of this operation will be written back to the same buffer.

        Parameters
        ----------
        req_info : dict
           request info from individual :class:`Worker`.

        req_info
        --------
        * *dtype* : str
           Numpy dtype of the shared memory buffer's elements.
        * *op* : str
           Reference name of a reduce operation type.
        * *shmem* : str
           Reference name to a shared memory buffer.

        .. versionadded:: 0.6.0

        """
        if not self._multinode:
            raise PlatoonError("Request to all_reduce, when multi-node is off.")
        if self._region_comm is None:
            raise PlatoonError("`all_reduce` request is not available. Check log.")
        dtype = req_info['dtype']
        op = req_info['op']
        array = self.shared_buffers[req_info['shmem']]
        try:
            mpi_op = op_to_mpi(op)
            mpi_dtype = dtype_to_mpi(dtype)
            self._region_comm.Allreduce([array, mpi_dtype], [array, mpi_dtype],
                                        op=mpi_op)
        except Exception as exc:
            raise PlatoonError("Failed to execute all_reduce across nodes on \
                               request: {}".format(req_info), exc)
        return ""  # So as to show that request type has been found

################################################################################
#                           Distribute Data Batches                            #
################################################################################

    def init_data(self, port, data_hwm=10):
        """
        Initialize the mini-batch socket.

        This must be called before using :meth:`send_mb`.

        Parameters
        ----------
        port : int
           The port to listen on.
        data_hwm : int
           High water mark, see the pyzmq docs.

        """
        self.acontext = zmq.Context()
        self.asocket = self.acontext.socket(zmq.PUSH)
        self.asocket.set_hwm(data_hwm)
        self.asocket.bind('tcp://*:{}'.format(port))

    def send_mb(self, arrays):
        """
        Send a mini-batch over the socket.

        This function may block if arrays are being sent faster than
        the clients can handle.

        Parameters
        ----------
        arrays : list of ndarrays
           List of :class:`numpy.ndarray` to send.  All arrays should be
           contiguous for better performance.

        """
        # The buffer protocol only works on contiguous arrays
        arrays = [numpy.ascontiguousarray(array) for array in arrays]
        headers = [numpy.lib.format.header_data_from_array_1_0(array)
                   for array in arrays]
        self.asocket.send_json(headers, zmq.SNDMORE)
        for array in arrays[:-1]:
            self.asocket.send(array, zmq.SNDMORE)
        self.asocket.send(arrays[-1])

################################################################################
#                               Helper functions                               #
################################################################################

    @staticmethod
    def get_workers_devices(devices):
        """
        This static method is used by a :class:`Controller` instance to
        determine which devices should it use in its :class:`Worker`s, in case
        it is not explicitly given in its arguments.

        :param devices: list of Theano device ids specified in arguments

        .. note::
           Devices fall to defaults, if they cannot be found with any
           available Platoon configuration method. Default devices are all CUDA
           devices in host.

        .. warning::
           If any devices cannot be inferred at all, process exits.

        .. versionadded:: 0.6.0

        """
        # Get :class:`Controller`'s devices
        import socket
        hostname = socket.gethostname()

        if devices:
            # 1. Use device names from arguments if they are specified
            devices_found = devices
        else:
            # 2. Try device names from configuration
            from platoon import configparser
            try:
                devices_found = configparser.fetch_devices_for_host(hostname)
            except KeyError:
                # 3. Else try to use all compatible GPUs in host
                try:
                    print("WARNING! Using all compatible GPUs in " + hostname + ".", file=sys.stderr)
                    from pygpu import gpuarray as ga
                    devcount = ga.count_devices("cuda", 0)
                    print("WARNING! Found {} GPUs!".format(devcount), file=sys.stderr)
                    devices_found = ["cuda" + str(i) for i in range(devcount)]
                except ImportError:
                    print("ERROR! Could not fetch devices for Controller.", file=sys.stderr)
                    sys.exit(2)
                except Exception as exc:
                    print("ERROR! pygpu: {}".format(exc), file=sys.stderr)
                    print("ERROR! Could not fetch devices for Controller.", file=sys.stderr)
                    sys.exit(2)

        if not devices_found:
            print("ERROR! Cound not find any compatible GPUs in " + hostname + ".", file=sys.stderr)
            sys.exit(4)

        print("## On " + hostname + " using: " + " ".join(devices_found))
        return devices_found

    @staticmethod
    def default_parser():
        """
        Returns base :class:`Controller`'s class parser for its arguments.

        This parser can be augmented with more arguments, if it is needed, in
        case a class which inherits :class:`Controller` exists.

        .. versionadded:: 0.6.0

        """
        parser = argparse.ArgumentParser(
            description="Base Platoon Controller process. Reigns over a computer node.")
        parser.add_argument('experiment_name', help='The name of your experiment. The launcher will expect to find the files <experiment_name>_worker.py and optionally <experiment_name>_controller.py.')
        parser.add_argument('log_directory', help='Directory where logging info and error files will exist.')
        single_or_multi = parser.add_mutually_exclusive_group(required=True)
        single_or_multi.add_argument('--single', action='store_true',
                                     help='Indicates that this Controller participates in a single-node platoon.')
        single_or_multi.add_argument('--multi', action='store_true',
                                     help='Indicates that this Controller participates in a multi-node platoon. Requires mpi4py')
        parser.add_argument('-D', '--devices', default=list(), nargs='+', type=str, metavar='devname',
                            required=False, help='List of device names (e.g. gpu0 or cuda1). Each device will be assigned to a separate worker. If this option is specified, experiment will be run in a single node.')
        parser.add_argument('-w', '--worker-args', required=False, help='The arguments that will be passed to your workers. (Ex: -w="learning_rate=0.1")')
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
        DEFAULT_KEYS = ['control_port', 'data_port', 'data_hwm',
                        'devices', 'experiment_name',
                        'log_directory', 'worker_args', 'multi']
        d = args.__dict__
        return dict((k, d[k]) for k in six.iterkeys(d) if k in DEFAULT_KEYS)


def spawn_controller():
    parser = Controller.default_parser()
    args = parser.parse_args()
    controller = Controller(**Controller.default_arguments(args))
    return controller.serve()

if __name__ == '__main__':
    rcode = spawn_controller()
    if rcode != 0:
        sys.exit(rcode)
