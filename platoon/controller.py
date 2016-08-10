# -*- coding: utf-8 -*-
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
    from mpi4py import MPI
except ImportError:
    MPI = None

from .util import (PlatoonError, mmap, launch_process,
                   op_to_mpi, dtype_to_mpi)


class Controller(object):
    """
    General multi-process controller

    This class provides the necessary features to dispatch data mini-batches
    to workers and handle control requests. Using this class should be done
    by having another class inherit from it and override the method
    `handle_control()`.

    .. warning::

        Due to the underlying implementation it is a bad idea to
        attempt to do both in the same process, even on different
        threads.  This will suffer from interlock problems and may
        negate any speedup you could get from using multiple Workers.

        Because of this issue, the class may be split in the future.

    Parameters
    ----------
    control_port : int
        The control port number.
    data_port : int
        The data port number.
    data_hwm : int
        High water mark (see pyzmq docs).
    experiment : tuple of 3 strings
        (experiment name, log directory, worker arguments)
    devices : list of strings
        Contains device names in clique order (prefer ring topology)
    multinode : bool
        True, if we start a multi-node experiment. Flag to start MPI.

    """

    def __init__(self, control_port=5567, data_port=None, data_hwm=10,
                 devices=None, workers=None, experiment_name='',
                 log_directory='', worker_args='', multi=False):
        self._workers = set()
        self._need_init = True

        self._devices = list()
        if devices is not None:
            self._devices = Controller.get_workers_devices(not multi, devices, workers)
        self._local_size = len(self._devices)
        self._global_size = self._local_size
        self._get_platoon_info_count = [0]
        self._init_new_shmem_count = [0]
        self._am_i_first_count = [0]

        signal.signal(signal.SIGTERM, self._handle_force_close)
        signal.signal(signal.SIGINT, self._handle_force_close)

        # If we are starting a multi-node training (new interface)
        self._multinode = multi
        if self._multinode:
            try:
                self._init_region_comm()
            except AttributeError as exc:
                print("WARNING! {} while being in multi-node mode".format(exc),
                      file=sys.stderr)

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

        # If we are using the new interface, then initialize workers
        if experiment_name:
            try:
                os.makedirs(log_directory)
            except OSError:
                pass
            try:
                for device in self._devices:
                    p = launch_process(log_directory, experiment_name,
                                       shlex.split(worker_args or ''), device,
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
                                  "behavior.".format(req, __class__.__name__))

    def _handle_base_control(self, req, worker_id, req_info):
        """
        This method handle base control commands.
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

    # Method `worker_is_done` is not supported anymore.
    # Rationale:
    # 1. For supporting multi-node controllers it was necessary to move the
    # spawning of worker process to Controller object, so Controller needs to
    # wait for the exit of its worker children. This was done for the following
    # reasons:
    #     a. controller processes may be MPI processes, while worker processes
    #     are not.
    #     b. In multi-node case, worker processes have to be spawned in a
    #     another host than the one executing the launcher.
    #     c. It would be better, if there was a uniform way to spawn worker
    #     process which is independent whether we are in the single-node or
    #     multi-node case.
    # 2. Given (1): Each process oughts to take care cleaning and exiting
    # gracefully by itself, unless a fatal error has happened or a irrecovable
    # error for the procedure has happened, in which case process should exit
    # normally with non-success code. In fatal or irrecovable cases, their
    # father process (i.e controller) must kill them (forcing them to exit
    # gracefully, if possible).
    #  def worker_is_done(self, worker_id):
    #      self._workers.discard(worker_id)
    #      self._should_stop = True

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
                            # error or an irrecovable fault
                            raise PlatoonError("A worker has exited with non-success code: {}".format(self._success))
                    else:  # other status changes are not desirable
                        raise PlatoonError("A worker has changed to a status other than exit.")
                try:
                    query = self.csocket.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:  # if a query has not happened, try again
                    continue
                except zmq.ZMQError as exc:
                    raise PlatoonError("while receiving using zmq socket", exc)

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
                    raise PlatoonError("while sending using zmq socket", exc)
        except PlatoonError as exc:  # if platoon fails kill all children workers
            print(exc, file=sys.stderr)
            self._clean()
        except Exception as exc:
            print(PlatoonError("Unexpected exception", exc), file=sys.stderr)
            self._clean()
        finally:  # Close sockets and unlink for shared memory
            self._close()
        return self._success

################################################################################
#                   Initialization and Finilization Methods                    #
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
        if MPI is None:
            raise AttributeError("mpi4py is not imported")
        self._region_comm = MPI.COMM_WORLD
        self._region_size = MPI.COMM_WORLD.Get_size()
        self._region_rank = MPI.COMM_WORLD.Get_rank()
        global_size = numpy.array([self._global_size])
        self._region_comm.Allreduce(global_size, global_size, op=MPI.SUM)
        self._global_size = global_size[0]

    def _clean(self):
        print("Cleaning up...", file=sys.stderr)
        self._kill_workers()
        if self._multinode:
            print("Aborting MPI job...", file=sys.stderr)
            self._region_comm.Abort(errorcode=1)

    def _kill_workers(self):
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
        print("Closing connections and unlinking memory...", file=sys.stderr)
        self.csocket.close()
        self.ccontext.term()
        if hasattr(self, 'asocket'):
            self.asocket.close()
            self.acontext.term()
        self._lock.close()
        try:
            self._lock.unlink()
        except posix_ipc.ExistentialError:
            pass
        for shmref in self._shmrefs.values():
            try:
                shmref.unlink()
            except posix_ipc.ExistentialError:
                pass

    def _handle_force_close(self, signum, frame):
        """Handle SIGTERM signals from MPI.Abort

        This is expected to happen when something abnormal has happened in other
        controllers over MPI.COMM_WORLD across host which participate in the
        multi-node training.

        """
        self._kill_workers()
        self._close()
        sys.exit(1)

################################################################################
#                               Control Requests                               #
################################################################################

    def _is_worker_first(self, counter):
        """Returns True, if in a mass request in a local platoon (workers in a
        single host) a worker's request reaches first its controller

        This will work only if every single worker participates successfully each
        time in a concurrent request of the same type to their controller.

        """
        if self._local_size == 1:
            return True
        counter[0] = (counter[0] + 1) % self._local_size
        if counter[0] == 1:
            return True
        return False

    def _get_platoon_info(self, req_info):
        first = self._is_worker_first(self._get_platoon_info_count)  # See :ref:`_is_worker_first`
        if first:
            self._local_id = "platoon-" + req_info['local_id']
        response = dict()
        response['local_id'] = self._local_id
        response['local_size'] = self._local_size
        response['local_rank'] = self._devices.index(req_info['device'])
        response['multinode'] = self._multinode
        response['global_size'] = self._global_size
        return response

    def _init_new_shmem(self, req_info):
        first = self._is_worker_first(self._init_new_shmem_count)  # See :ref:`_is_worker_first`
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
        if not self._multinode:
            raise PlatoonError("Request to all_reduce, when multi-node is off.")
        if MPI is None:
            raise AttributeError("mpi4py is not imported.")
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

    def init_data(self, port, hwm=10):
        """
        Initialize the mini-batch socket.

        This must be called before using :meth:`send_mb`.

        Parameters
        ----------
        port : int
            The port to listen on.
        hwm : int
            High water mark, see the pyzmq docs.

        """
        self.acontext = zmq.Context()
        self.asocket = self.acontext.socket(zmq.PUSH)
        self.asocket.set_hwm(hwm)
        self.asocket.bind('tcp://*:{}'.format(port))

    def send_mb(self, arrays):
        """
        Send a mini-batch over the socket.

        This function may block if arrays are being sent faster than
        the clients can handle.

        Parameters
        ----------
        arrays : list of ndarrays
            List of numpy.ndarray to send.  All arrays should be
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
    def get_workers_devices(single, devices, workers):
        # Get Controller's devices
        import socket
        hostname = socket.gethostname()

        if single and devices:
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
                    print("WARNING! Using all compatible GPUs in host.", file=sys.stderr)
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
            print("ERROR! Cound not find any compatible GPUs in host.", file=sys.stderr)
            sys.exit(4)

        if workers:
            if workers > len(devices_found):
                print("WARNING! Given {0} workers but {1} given devices. Using {1} workers.".format(workers, len(devices)),
                      file=sys.stderr)
                workers_found = len(devices_found)
            else:
                workers_found = workers
        else:
            workers_found = len(devices_found)

        final_devices = devices_found[:workers_found]
        print("## On " + hostname + " using: " + " ".join(final_devices))
        return final_devices

    @staticmethod
    def default_parser():
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
                            required=False, help='List of Theano device names (e.g. gpu0 or cuda1). Each device will be assigned to a separate worker. If this option is specified, experiment will be run in a single node.')
        parser.add_argument('-nw', '--workers', type=int, metavar='num_of_workers',
                            required=False, help='Number of workers spawned by this controller for this host.')
        parser.add_argument('-w', '--worker-args', required=False, help='The arguments that will be passed to your workers. (Ex: -w="learning_rate=0.1")')
        parser.add_argument('--control-port', default=5567, type=int, required=False, help='The control port number.')
        parser.add_argument('--data-port', type=int, required=False, help='The data port number.')
        parser.add_argument('--data-hwm', default=10, type=int, required=False, help='The data port high water mark')

        return parser

    @staticmethod
    def default_arguments(args):
        DEFAULT_KEYS = ['control_port', 'data_port', 'data_hwm',
                        'devices', 'workers', 'experiment_name',
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
