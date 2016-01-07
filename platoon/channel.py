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


_ffi = cffi.FFI()
_ffi.cdef("""
void *mmap(void *, size_t, int, int, int, size_t);
""")
_lib = _ffi.dlopen(None)


def _mmap(addr=_ffi.NULL, length=0, prot=0x3, flags=0x1, fd=0, offset=0):
    m = _lib.mmap(addr, length, prot, flags, fd, offset)
    if m == _ffi.cast('void *', -1):
        raise OSError(_ffi.errno, "for mmap")
    return _ffi.buffer(m, length)


class Controller(object):
    """
    Abstract multi-process controller

    This class provides the necessary features to dispatch data minibatches
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
    cport : int
        The control port number.
    hwm : int
        High water mark (see pyzmq docs).

    """

    def __init__(self, port=None, cport=None, hwm=10):

        self._should_stop = False
        self._worker_list = set()

        if port:
            self.init_data(port, hwm)
        if cport:
            self.init_control(cport)

    def init_data(self, port, hwm=10):
        """
        Initialize the minibatch socket.

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

    def init_control(self, port):
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
        Send a minibatch over the socket.

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

            response = self.handle_control(query['req'], query['worker_id'])

            self.csocket.send(json.dumps(response))
        self.csocket.close()


def descr_size(dtype, shape):
    size = dtype.itemsize
    for s in shape:
        size *= s
    return size


class Worker(object):
    """
    Worker object. Each worker should have one instance of this class.

    This class handles the communication/synchronization with other processes.
    The features to do so (control channel, minibatch channel and shared
    parameters) are all independent and optional so you don't have to use all
    of them.

    Parameters
    ----------
    port : int, optional
        Will call :meth:`init_mb_sock` with this port.
    cport : int
        Will call :meth:`init_control_sock` with this port.
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

    def __init__(self, port=None, cport=None, socket_timeout=300000, hwm=10):
        self.context = zmq.Context()

        self._socket_timeout = socket_timeout

        self._worker_id = os.getpid()

        if port:
            self.init_mb_sock(port, hwm)
        if cport:
            self.init_control_sock(cport)

    def init_mb_sock(self, port, hwm=10):
        """
        Initialize the minibatch socket.

        This must be called before using :meth:`recv_mb`.

        Parameters
        ----------
        port : int
            The port to reach the minibatch server on.
        hwm : int
            High water mark, see pyzmq docs.

        """
        self.asocket = self.context.socket(zmq.PULL)
        self.asocket.setsockopt(zmq.LINGER, 0)
        self.asocket.set_hwm(hwm)
        self.asocket.connect("tcp://localhost:{}".format(port))

        self.apoller = zmq.Poller()
        self.apoller.register(self.asocket, zmq.POLLIN)

    def init_control_sock(self, port):
        """
        Intialize control socket.

        This must be called before using :meth:`send_req`.

        Paramters
        ---------
        port : int
            Port where the control master is listening on.

        """
        self.csocket = self.context.socket(zmq.REQ)
        self.csocket.setsockopt(zmq.LINGER, 0)
        self.csocket.connect('tcp://localhost:{}'.format(port))

        self.cpoller = zmq.Poller()
        self.cpoller.register(self.csocket, zmq.POLLIN)

    def init_shared_params(self, job_name, params, param_sync_rule,
                           cleanup=False):
        """
        Intialize shared memory parameters.

        This must be called before accessing the params attribute
        and/or calling :meth:`sync_params`, :meth:`lock_params` or
        :meth:`unlock_params`.

        Paramters
        ---------
        job_name : str
            An identifier.  This must be the same across all Workers
            that share paramters.
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
        if cleanup:
            try:
                posix_ipc.unlink_semaphore(job_name+'lock')
            except posix_ipc.ExistentialError:
                pass
        self.lock = posix_ipc.Semaphore(job_name+'lock', posix_ipc.O_CREAT,
                                        initial_value=1)

        params_descr = [(numpy.dtype(p.dtype), p.get_value(borrow=True).shape)
                        for p in params]
        params_size = sum(descr_size(*d) for d in params_descr)
        if cleanup:
            try:
                posix_ipc.unlink_shared_memory(job_name+'params')
            except posix_ipc.ExistentialError:
                pass
            self._shmref = posix_ipc.SharedMemory(job_name+'params',
                                                  posix_ipc.O_CREAT,
                                                  size=params_size)
        self._shmref = posix_ipc.SharedMemory(job_name+'params')
        self._shm = _mmap(fd=self._shmref.fd, length=params_size)
        self._shmref.close_fd()
        self.shared_params = []
        off = 0
        for dtype, shape in params_descr:
            self.shared_params.append(numpy.ndarray(shape, dtype=dtype,
                                                    buffer=self._shm,
                                                    offset=off))
            off += descr_size(dtype, shape)

    def recv_mb(self):
        """
        Recieve a minibatch for processing.

        A minibatch is composed of a number of numpy arrays.

        Returns
        -------
        list
            The list of numpy arrays for the minibatch

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
            timeout of 0 will raise an error immediatly if the lock is
            not available.

        """
        self.lock.acquire(timeout)

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
        self.lock.release()

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
