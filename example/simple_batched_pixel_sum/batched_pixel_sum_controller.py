from __future__ import print_function
import os
import sys
import gzip
import time
import six
from six.moves import cPickle
from multiprocessing import Process

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from platoon.channel import Controller


class BatchedPixelSumController(Controller):

    def __init__(self, control_port, batch_port, dataset, batch_size):
        Controller.__init__(self, control_port, None)
        # The data socket should be initialized in the process that will handle
        # the batch.
        # That is why it's not initialized in the parent constructor. Second
        # param = None
        self._batch_port = batch_port

        self._start_time = None
        self._should_stop = False

        self._batch_size = batch_size
        self._dataset = dataset

        self._nb_batch_processed = 0
        self._nb_batch_to_process = (dataset.shape[0] // batch_size)

    def start_batch_server(self):
        self.p = Process(target=self._send_mb)
        self.p.start()

    def _send_mb(self):
        self.init_data(self._batch_port)

        for i in range(self._dataset.shape[0] // self._batch_size):
            batch_start = i*self._batch_size
            batch_stop = (i + 1)*self._batch_size
            self.send_mb(self._dataset[batch_start:batch_stop])

        self.asocket.close()
        print("Done Sending MB.")

        # TODO: Find a solution for this
        # Sleeping to give the chance to the worker to empty the queue before
        # the MB process dies
        import time
        time.sleep(2)

    def handle_control(self, req, worker_id):
        print("# Handling req: {}".format(req))
        control_response = ''

        if req == 'next':
            # Start a global execution timer
            if self._start_time is None:
                self._start_time = time.time()
            control_response = 'train'

        elif req == 'get_data_shape':
            control_response = self._dataset[0].shape

        elif 'done' in req:
            self._nb_batch_processed += req['done']
            print("{} batches processed by worker so far."
                  .format(self._nb_batch_processed))

        if self._nb_batch_processed == self._nb_batch_to_process:
            control_response = 'stop'
            self.worker_is_done(worker_id)
            print("Training time {0:.4f}s"
                  .format(time.time() - self._start_time))
        return control_response


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_port', default=5566, type=int, required=False,
                        help='Port on which the batches will be transfered.')
    parser.add_argument('--control_port', default=5567, type=int,
                        required=False, help='Port on which the control '
                        'commands will be sent.')
    parser.add_argument('--batch-size', default=1000, type=int, required=False,
                        help='Size of the batches.')

    return parser.parse_args()


def get_mnist(path):
    import os
    import urllib

    if not os.path.exists(path):
        print("Downloading mnist ...", end=' ')
        url = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"

        urllib.urlretrieve(url, path)
        print("Done")


if __name__ == '__main__':
    args = parse_arguments()

    mnist_path = "../data/mnist.pkl.gz"

    get_mnist(mnist_path)

    with gzip.open(mnist_path, 'rb') as f:
        kwargs = {}
        if six.PY3:
            kwargs['encoding'] = 'latin1'
        train_set, _, _ = cPickle.load(f, **kwargs)

    controller = BatchedPixelSumController(control_port=args.control_port,
                                           batch_port=args.batch_port,
                                           dataset=train_set[0],
                                           batch_size=args.batch_size)
    controller.start_batch_server()
    controller.serve()
