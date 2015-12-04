from multiprocessing import Process
import cffi

import zmq
import posix_ipc

# You need:
# $ conda install pyzmq cffi
# $ pip install posix_ipc


_ffi = cffi.FFI()
_ffi.cdef("""
void *mmap(void *, size_t, int, int, int, size_t);
""")
_lib = _ffi.dlopen(None)


def _mmap(addr=_ffi.NULL, length=0, prot=0x3, flags=0, fd=0, offset=0):
    m = _lib.mmap(addr, length, prot, flags, fd, offset)
    if m == -1:
        raise OSError("could not mmap")
    return _ffi.buffer(m, length)


class Controller(object):
    """
    Controller object

    This may be badly named because it only handles sending
    minibatches to the worker processes.

    Parameters
    ----------
    port : int
        The port number to communicate over
    hwm : int
        High water mark (see pyzmq docs).
    """
    def __init__(self, port, hwm=10):
        context = zmq.Context()
        self.asocket = context.socket(zmq.PUSH)
        self.asocket.set_hwm(hwm)
        self.asocket.bind('tcp://*:{}'.format(port))

    def send_mb(self, arrays):
        # The buffer protocol only works on contiguous arrays
        arrays = [numpy.ascontiguousarray(array) for array in arrays]
        headers = [header_data_from_array_1_0(array) for array in arrays]
        self.asocket.send_json(headers, zmq.SNDMORE)
        for array in arrays[:-1]:
            self.asocket.send(array, zmq.SNDMORE)
        self.asocket.send(arrays[-1])


def descr_size(dtype, shape):
    size = dtype.itemsize
    for s in shape:
        size *= s
    return s


class Worker(object):
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
    hwm : int
        High water mark (see pyzmq docs).
    """
    def __init__(self, job_name, params_descr, port, hwm=10):
        context = zmq.Context()
        self.asocket = context.socket(zmq.PULL)
        self.asocket.set_hwm(hwm)
        self.asocket.connect("tcp://localhost:{}".format(port))

        self.lock = posix_ipc.Semaphore(job_name+'lock', posix_ipc.O_CREAT,
                                        initial_value=1)
        params_size = sum(descr_size(*d) for d in params_descr)
        self._shmref = posix_ipc.SharedMemory(job_name+'params',
                                              posix_ipc.O_CREAT,
                                              size=params_size)
        f = os.fd
        self._shm = mmap.mmap(self._shmref.fd, params_size)
        self._shmref.close_fd()
        self.params = []
        s = 0
        e = 0
        for dtype, shape in params_descr:
            e += descr_size(dtype, shape)
            

    def recv_mb(self):
        headers = self.asocket.recv_json()
        arrays = []
        for header in headers:
            data = socket.recv()
            buf = buffer_(data)
            array = numpy.frombuffer(buf, dtype=numpy.dtype(header['descr']))
            array.shape = header['shape']
            if header['fortran_order']:
                array.shape = header['shape'][::-1]
                array = array.transpose()
            arrays.append(array)
        return arrays

    def lock_params(self):
        self.lock.aquire()

    def unlock_params(self):
        self.lock.release()
