from threading import Thread
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


class Lieutenant(object):
    """
    Controller object

    This may be badly named because it only handles sending
    minibatches to the worker processes.

    Parameters
    ----------
    port : int
        The port number to communicate over
    cport : int
        The control port number.
    hwm : int
        High water mark (see pyzmq docs).

    """
    def __init__(self, port, cport, hwm=10):
        context = zmq.Context()
        self.asocket = context.socket(zmq.PUSH)
        self.asocket.set_hwm(hwm)
        self.asocket.bind('tcp://*:{}'.format(port))
        self.csocket = context.socket(zmq.REP)
        self.csocket.bind('tcp://*:{}'.format(cport))

    def send_mb(self, arrays):
        # The buffer protocol only works on contiguous arrays
        arrays = [numpy.ascontiguousarray(array) for array in arrays]
        headers = [numpy.lib.format.header_data_from_array_1_0(array)
                   for array in arrays]
        self.asocket.send_json(headers, zmq.SNDMORE)
        for array in arrays[:-1]:
            self.asocket.send(array, zmq.SNDMORE)
        self.asocket.send(arrays[-1])

    def handle_control(self, req):
        return 'error! override handle_control on the Lieutenant.'

    def serve(self):
        while True:
            req = self.csocket.recv()
            rep = self.handle_control(json.loads(req))
            self.csocket.send(json.dumps(rep))


def descr_size(dtype, shape):
    size = dtype.itemsize
    for s in shape:
        size *= s
    return size


class Soldier(object):
    """
    Worker object

    Each worker should have one instance of this object.

    Parameters
    ----------
    job_name : str
        Some name that is the same for all workers in the same job.
        This should differ from other jobs on the same machine.
    params_descr : list of (dtype, shape)
        Describes the paramters for the job.  This is required for the
        total size and the mapping between the shared memory and the
        viewing arrays.
    port : int
        This should be the same number as the port for the Controller
    cport : int
        Control port number from the controller.
    hwm : int
        High water mark (see pyzmq docs).

    Attributes
    ----------
    params : list of ndarrays
        This will have numpy ndarray in the same order as params_descr.
        These arrays are backed by shared memory.
    """
    def __init__(self, job_name=None, params_descr=None,
                 port=None, cport=None, hwm=10):
        self.context = zmq.Context()
        if port:
            self.init_mb_sock(port, hwm)
        if cport:
            self.init_control_sock(cport)
        if job_name and params_descr:
            self.init_shared_params(job_name, params_descr)

    def init_mb_sock(self, port, hwm=10):
        self.asocket = self.context.socket(zmq.PULL)
        self.asocket.set_hwm(hwm)
        self.asocket.connect("tcp://localhost:{}".format(port))

    def init_control_sock(self, port):
        self.csocket = self.context.socket(zmq.REQ)
        self.csocket.connect('tcp://localhost:{}'.format(port))

    def init_shared_params(self, job_name, params, param_sync_rule,
                           cleanup=False):
        self.local_params = params
        self.param_sync_rule = param_sync_rule
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
        headers = self.asocket.recv_json()
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

    def lock_params(self):
        """
        Lock the shared params across all workers.

        This is advisory and does not prevent concurrent access.
        """
        self.lock.acquire()

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
        """
        if synchronous:
            self.lock_params()
        
        # Read the values of the local parameters, update them and the
        # shared parameters and write back the new values of the local
        # parameters
        local_param_values = [p.get_value() for p in self.local_params]
        self.param_sync_rule.update_params(local_param_values,
                                           self.shared_params)        
        for param, value in zip(self.local_params, local_param_values):
            param.set_value(value)
        
        if synchronous:
            self.unlock_params()

    def send_req(self, req):
        """
        Send a control request.

        Parameters
        ----------
        req : object
            This is a json-encodable object that will be sent to the
            lieutenant.

        Returns
        -------
        object
            Json-decoded object

        """
        self.csocket.send(json.dumps(req))
        return json.loads(self.csocket.recv())
