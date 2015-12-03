from multiprocessing import Process, Pipe
from threading import Thread
import zmq
import posix_ipc

# You need:
# $ conda install pyzmq
# $ pip install posix_ipc


class Controller(object):
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


class Worker(object):
    def __init__(self, job_name, params_size, port, hwm=10):
        context = zmq.Context()
        self.asocket = context.socket(zmq.PULL)
        self.asocket.set_hwm(hwm)
        self.asocket.connect("tcp://localhost:{}".format(port))

        self.lock = posix_ipc.Semaphore(job_name+'lock', posix_ipc.O_CREAT,
                                        initial_value=1)
        self.shm = posix_ipc.SharedMemory(job_name+'params', posix_ipc.O_CREAT,
                                          size=params_size)

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
