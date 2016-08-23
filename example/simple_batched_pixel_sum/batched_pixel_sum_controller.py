from __future__ import absolute_import, print_function
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

    def __init__(self, batch_port, dataset, batch_size, default_args):
        super(BatchedPixelSumController, self).__init__(**default_args)
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
            batch_start = i * self._batch_size
            batch_stop = (i + 1) * self._batch_size
            self.send_mb(self._dataset[batch_start:batch_stop])

        self.asocket.close()
        print("Done Sending MB.")

        # TODO: Find a solution for this
        # Sleeping to give the chance to the worker to empty the queue before
        # the MB process dies
        import time
        time.sleep(2)

    def handle_control(self, req, worker_id, req_info):
        print("# Handling req: {}".format(req))
        control_response = ''

        if req == 'next':
            if not self._should_stop:
                # Start a global execution timer
                if self._start_time is None:
                    self._start_time = time.time()
                control_response = 'train'
            else:
                control_response = 'stop'
        elif req == 'get_data_shape':
            control_response = self._dataset[0].shape
        elif req == 'done':
            self._nb_batch_processed += req_info['num_batches']
            print("{} batches processed by worker so far."
                  .format(self._nb_batch_processed))

        if self._nb_batch_processed >= self._nb_batch_to_process:
            if not self._should_stop:
                print("Training time {:.4f}s".format(
                    time.time() - self._start_time))
            self._should_stop = True

        return control_response


def parse_arguments():
    parser = Controller.default_parser()
    parser.add_argument('--batch_port', default=5566, type=int, required=False,
                        help='Port on which the batches will be transfered.')
    parser.add_argument('--batch-size', default=1000, type=int, required=False,
                        help='Size of the batches.')

    return parser.parse_args()


def get_mnist(path):
    import os
    from six.moves import urllib

    if not os.path.exists(path):
        print("Downloading mnist ...", end=' ')
        url = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"

        urllib.request.urlretrieve(url, path)
        print("Done")


def spawn_controller():
    args = parse_arguments()

    mnist_path = "../data/mnist.pkl.gz"

    get_mnist(mnist_path)

    with gzip.open(mnist_path, 'rb') as f:
        kwargs = {}
        if six.PY3:
            kwargs['encoding'] = 'latin1'
        train_set, _, _ = cPickle.load(f, **kwargs)

    controller = BatchedPixelSumController(batch_port=args.batch_port,
                                           dataset=train_set[0],
                                           batch_size=args.batch_size,
                                           default_args=Controller.default_arguments(args))
    controller.start_batch_server()
    return controller.serve()

if __name__ == '__main__':
    rcode = spawn_controller()
    if rcode != 0:
        sys.exit(rcode)
