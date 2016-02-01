import os
import numpy
import json

import cffi
import zmq
import posix_ipc

# You need:
# $ conda install pyzmq cffi
# $ pip install posix_ipc

# Also this was only tested in python 2.7


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
    hwm : int
        High water mark (see pyzmq docs).

    """

    def __init__(self, control_port, port=None, hwm=10):

        self._should_stop = False
        self._worker_list = set()
        self._need_init = True

        if port:
            self.init_data(port, hwm)

        self._init_control_socket(control_port)

        ## Cleanup and init global lock and job_uid name ##
        self._job_uid = "platoon_{0}_{1}".format(os.path.basename(os.path.expanduser('~')), control_port)

        self._lock_name = "{}lock".format(self._job_uid)
        # The ExistentialError is apparently the only way to verify if the semaphore/shared_memory exists.
        try:
            posix_ipc.unlink_semaphore(self._lock_name)
        except posix_ipc.ExistentialError:
            pass
        # Initializing lock
        posix_ipc.Semaphore(self._lock_name, posix_ipc.O_CREAT, initial_value=1)

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

    def handle_control(self, req, worker_id):
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

    def _handle_base_control(self, req, worker_id):
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

        return response

    def worker_is_done(self, worker_id):
        self._worker_list.discard(worker_id)
        self._should_stop = True

    def serve(self):
        """
        This method will handle control messages until the should_stop flag
        has been raised and that all the known worker are done.
        """

        while (not self._should_stop) or self._worker_list:
            query = json.loads(self.csocket.recv())
            self._worker_list.add(query['worker_id'])

            response = self._handle_base_control(query['req'], query['worker_id'])
            if response is None:
                response = self.handle_control(query['req'], query['worker_id'])

            self.csocket.send(json.dumps(response))
        self.csocket.close()


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

    def __init__(self, control_port, port=None, socket_timeout=300000, hwm=10):
        self.context = zmq.Context()

        self._socket_timeout = socket_timeout

        self._worker_id = os.getpid()

        if port:
            self.init_mb_sock(port, hwm)

        self._init_control_socket(control_port)

        self._job_uid = self.send_req("platoon-get_job_uid")
        self._lock = posix_ipc.Semaphore("{}lock".format(self._job_uid))

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

    @staticmethod
    def _mmap(length=0, prot=0x3, flags=0x1, fd=0, offset=0):
        _ffi = cffi.FFI()
        _ffi.cdef("void *mmap(void *, size_t, int, int, int, size_t);")
        _lib = _ffi.dlopen(None)

        addr = _ffi.NULL

        m = _lib.mmap(addr, length, prot, flags, fd, offset)
        if m == _ffi.cast('void *', -1):
            raise OSError(_ffi.errno, "for mmap")
        return _ffi.buffer(m, length)

    def _get_descr_size(self, dtype, shape):
        size = dtype.itemsize
        for s in shape:
            size *= s
        return size

    def init_shared_params(self, params, param_sync_rule):
        """
        Initialize shared memory parameters.

        This must be called before accessing the params attribute
        and/or calling :meth:`sync_params`, :meth:`lock_params` or
        :meth:`unlock_params`.

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

        params_descr = [(numpy.dtype(p.dtype), p.get_value(borrow=True).shape) for p in params]
        params_size = sum(self._get_descr_size(*d) for d in params_descr)

        shared_mem_name = "{}_params".format(self._job_uid)

        # Acquire lock to decide who will init the shared memory
        self.lock_params()

        need_init = self.send_req("platoon-need_init")
        if need_init:
            # The ExistentialError is apparently the only way to verify if the shared_memory exists.
            try:
                posix_ipc.unlink_shared_memory(shared_mem_name)
            except posix_ipc.ExistentialError:
                pass

            self._shmref = posix_ipc.SharedMemory(shared_mem_name, posix_ipc.O_CREAT, size=params_size)
        else:
            self._shmref = posix_ipc.SharedMemory(shared_mem_name)

        self._shm = self._mmap(fd=self._shmref.fd, length=params_size)
        self._shmref.close_fd()
        self.shared_params = []
        off = 0

        for dtype, shape in params_descr:
            self.shared_params.append(numpy.ndarray(shape, dtype=dtype, buffer=self._shm, offset=off))
            off += self._get_descr_size(dtype, shape)

        if need_init:
            self.copy_to_global(synchronous=False)

        self.unlock_params()

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

            buf = buffer(data)
            array = numpy.ndarray(
                buffer=buf, shape=header['shape'],
                dtype=numpy.dtype(header['descr']),
                order='F' if header['fortran_order'] else 'C')
            arrays.append(array)
        return arrays

    def lock_params(self, timeout=None):
        """
        Lock the shared params across all workers.

        This is advisory and does not prevent concurrent access.

        Parameters
        ----------
        timeout : int
            Amount of time to wait for the lock to be available.  A
            timeout of 0 will raise an error immediately if the lock is
            not available.

        """
        self._lock.acquire(timeout)

    def unlock_params(self):
        """
        Unlock the shared params across all workers.

        The current implementation does not ensure that the process
        that locked the params is the one that unlocks them.  It also
        does not prevent one process from unlocking more than once
        (which will allow more than one process to hold the lock).

        Make sure you follow proper lock/unlock logic in your program
        to avoid these problems.
        """
        self._lock.release()

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
            self.lock_params()

        self.update_fn(self.shared_params)

        if synchronous:
            self.unlock_params()

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
            self.lock_params()

        for p, v in zip(self.local_params, self.shared_params):
            p.set_value(v)

        if synchronous:
            self.unlock_params()

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
            self.lock_params()

        for p, v in zip(self.local_params, self.shared_params):
            v[:] = p.get_value(borrow=True)

        if synchronous:
            self.unlock_params()

    def send_req(self, req):
        """
        Send a control request.

        Parameters
        ----------
        req : object
            This is a json-encodable object that will be sent to the
            Controller.

        Returns
        -------
        object
            Json-decoded object

        """
        query = {"worker_id": self._worker_id, "req": req}
        self.csocket.send(json.dumps(query))

        socks = dict(self.cpoller.poll(self._socket_timeout))
        if socks and socks.get(self.csocket) == zmq.POLLIN:
            return json.loads(self.csocket.recv())
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
