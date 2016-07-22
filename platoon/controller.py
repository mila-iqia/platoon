import os

import numpy
import posix_ipc
import zmq
from mpi4py import MPI

from util import mmap


class Controller(object):
    """
    Abstract multi-process controller

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
    port : int
        The port number to communicate over
    control_port : int
        The control port number.
    device_list : list of strings
        Contains device names in clique order (prefer ring topology)
    hwm : int
        High water mark (see pyzmq docs).

    """

    def __init__(self, control_port, device_list, global_size, global_rank,
                 port=None, hwm=10, global_comm_id=""):

        self._should_stop = False
        self._worker_list = set()
        self._need_init = True

        self._device_list = device_list
        # Controllers' MPI comm info
        self._global_rank = global_rank
        self._global_size = global_size
        self._global_comm_id = "platoon-global_comm_" + global_comm_id
        self._global_comm = None
        # These 3 variables below will be used to take advantage of multi-node
        # support of NCCL (later)
        self._local_size = 0
        self._count_workers = 0
        self._controller_rank = 0
        self._region_size = 0

        if port:
            self.init_data(port, hwm)

        self._init_control_socket(control_port)

        # Cleanup and init global lock and job_uid name ##
        self._job_uid = "platoon_{0}_{1}".format(
            os.path.basename(os.path.expanduser('~')), control_port)

        self._lock_name = "{}lock".format(self._job_uid)
        # The ExistentialError is apparently the only way to verify if
        # the semaphore/shared_memory exists.
        try:
            posix_ipc.unlink_semaphore(self._lock_name)
        except posix_ipc.ExistentialError:
            pass
        # Initializing lock
        self._lock = posix_ipc.Semaphore(self._lock_name, posix_ipc.O_CREAT,
                                         initial_value=1)

        self._shmrefs = dict()
        self.shared_buffers = dict()

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
        acontext = zmq.Context()
        self.asocket = acontext.socket(zmq.PUSH)
        self.asocket.set_hwm(hwm)
        self.asocket.bind('tcp://*:{}'.format(port))

    def _init_control_socket(self, port):
        """
        Initialize the control socket.

        This must be called before using :meth:`serve`.

        Parameters
        ----------
        port : int
            The port to listen on.

        """
        ccontext = zmq.Context()
        self.csocket = ccontext.socket(zmq.REP)
        self.csocket.bind('tcp://*:{}'.format(port))

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

    def handle_control(self, req, worker_id, req_info):
        """
        Re-implement or assign a handler to this function to do
        something with control messages.

        The replacement get one parameter which is the request and
        should return the response which must be a json-encodable
        object. Other code is responsible for handling decoding,
        encoding and the network.

        """
        raise NotImplementedError("The Controller class should not be "
                                  "instantiated directly. Classes that "
                                  "inherit from Controller should override "
                                  "the method `handle_control()`")

    def _handle_base_control(self, req, worker_id, req_info):
        """
        This method handle base control commands.
        Those commands should not be used in the handle_control method.
        All base control commands should start with "platoon-".
        """
        response = None
        if req == "platoon-get_job_uid":
            self._local_size += 1
            response = self._job_uid

        elif req == "platoon-get_device":
            response = self._device_list[req_info['local_rank']]

        elif req == "platoon-need_init":
            response = self._need_init
            self._need_init = False

        elif req == "platoon-get_region_info":
            # Init MPI and decide for a Platoon region on which Controller rules
            if self._global_comm is None:
                self._region_id = b"platoon-" + req_info['region_id']
                self._region_size = self._local_size
                self._init_global_comm()  # May change region of workers to be multi-node
            response = dict()
            response['region_id'] = self._region_id
            response['region_size'] = self._region_size
            response['regional_rank'] = self._controller_rank + req_info['local_rank']

        elif req == "platoon-init_new_shmem":
            first = self.is_worker_first()  # See :ref:is_worker_first
            if first:
                self._last_shmem_name = "platoon-{0}_{1}_buffer".format(self._job_uid,
                                                                        len(self.shared_buffers))
                try:
                    posix_ipc.unlink_shared_memory(self._last_shmem_name)
                except posix_ipc.ExistentialError:
                    pass

                size = req_info['size']
                self._last_shmref = posix_ipc.SharedMemory(self._last_shmem_name,
                                                           posix_ipc.O_CREAT,
                                                           size=size)
                self._last_shm = mmap(fd=self._last_shmref.fd, length=size)
                self._last_shmref.close_fd()
                # We want every worker to get the same shared memory name that is
                # was declared in the first call of a mass request to this
                # controller for initializing a new shared memory.
                self._shmrefs[self._last_shmem_name] = self._last_shmref
                # Keep for unlinking when closing
                self.shared_buffers[self._last_shmem_name] = self._last_shm
            response = self._last_shmem_name

        elif req == "platoon-am_i_first":
            response = self.is_worker_first()  # See :ref:is_worker_first

        elif req == "platoon-all_reduce":
            dtype = req_info['dtype']
            op = req_info['op']
            array = self.shared_buffers[req_info['shmem']]
            #  mpi_dtype = dtype_to_mpi(dtype)  # TODO
            #  mpi_op = op_to_mpi(op)  # TODO
            self._global_comm.Allreduce([array, mpi_dtype], [array, mpi_dtype],
                                        op=mpi_op)
            # TODO add try/raise/finally and respond with success or failure to
            # worker

        return response

    def is_worker_first(self):
        """Returns True, if in a mass request in a local platoon (workers in a
        single host) a worker's request reaches first its controller

        This will work only if every single worker participates successfully each
        time in a concurrent request of the same type to their controller.
        """
        self._count_workers = (self._count_workers + 1) % self._local_size
        if self._count_workers == 1:
            return True
        return False

    def worker_is_done(self, worker_id):
        self._worker_list.discard(worker_id)
        self._should_stop = True

    def serve(self):
        """
        This method will handle control messages until the should_stop flag
        has been raised and that all the known worker are done.
        """

        while (not self._should_stop) or self._worker_list:
            query = self.csocket.recv_json()
            self._worker_list.add(query['worker_id'])

            response = self._handle_base_control(query['req'],
                                                 query['worker_id'],
                                                 query['req_info'])
            if response is None:
                response = self.handle_control(query['req'],
                                               query['worker_id'],
                                               query['req_info'])

            self.csocket.send_json(response)
        self.csocket.close()

    def _init_global_comm(self):
        # TODO
        pass
