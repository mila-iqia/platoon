import gzip
import cPickle
import numpy as np
from numpy.testing import assert_allclose
import theano
from theano import tensor as T
from theano.compat.python2x import OrderedDict


class BatchedPixelSum(object):

    def __init__(self, dataset, batch_size):
        self._batch_size = batch_size
        self._dataset = dataset

        self._computed_sum = theano.shared(value=np.zeros(dataset.shape[1], dtype=theano.config.floatX), name='sum', borrow=True)

        input = T.matrix(dtype=theano.config.floatX)
        batch_sum = T.sum(input, axis=0, dtype=theano.config.floatX)

        updates = OrderedDict()
        updates[self._computed_sum] = (self._computed_sum + batch_sum)

        self._update_sum = theano.function(name='learn',
                                           inputs=[input],
                                           updates=updates)

    def get_sum(self):
        for i in xrange(self._dataset.shape[0]/self._batch_size):
            batch_start = i*self._batch_size
            batch_stop = (i + 1)*self._batch_size
            print "Summing from {} to {}.".format(batch_start, batch_stop)
            self._update_sum(self._dataset[batch_start:batch_stop])
        return self._computed_sum.get_value()


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1000, type=int, required=False, help='Size of the batches.')

    return parser.parse_args()


def get_mnist(path):
    import os
    import urllib

    if not os.path.exists(path):
        print "Downloading mnist ...",
        url = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"

        urllib.urlretrieve(url, path)
        print "Done"

if __name__ == '__main__':
    args = parse_arguments()

    mnist_path = "../data/mnist.pkl.gz"

    get_mnist(mnist_path)

    with gzip.open(mnist_path, 'rb') as f:
        train_set, _, _ = cPickle.load(f)

    bps = BatchedPixelSum(train_set[0], args.batch_size)

    computed_sum = bps.get_sum()

    # Get actual answer for testing
    real_sum = train_set[0].sum(axis=0, dtype=theano.config.floatX)
    assert_allclose(computed_sum, real_sum)
