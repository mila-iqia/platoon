from __future__ import print_function
import os
import sys
import gzip
import cPickle

import numpy as np
from numpy.testing import assert_allclose

import theano
from theano import tensor as T
from theano.compat.python2x import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from platoon.channel import Worker
from platoon.param_sync import ParamSyncRule


class SUMSync(ParamSyncRule):

    def update_params(self, local_params, master_params):
        """
        Update the master params and reset to local params.
        """
        master_params[0] += local_params[0]
        local_params[0].fill(0)


class BatchedPixelSum(object):

    def __init__(self, control_port, batch_port):
        self._worker = Worker(control_port=control_port, port=batch_port)

        data_shape = self._worker.send_req('get_data_shape')

        self._computed_sum = theano.shared(value=np.zeros(data_shape, dtype=theano.config.floatX), name='sum', borrow=True)

        self._worker.init_shared_params(params=[self._computed_sum], param_sync_rule=SUMSync())

        input = T.matrix(dtype=theano.config.floatX)
        batch_sum = T.sum(input, axis=0, dtype=theano.config.floatX)

        updates = OrderedDict()
        updates[self._computed_sum] = (self._computed_sum + batch_sum)

        self._update_sum = theano.function(name='learn',
                                           inputs=[input],
                                           updates=updates)

    def get_sum(self):
        nb_batches_before_sync = 10

        while True:
            step = self._worker.send_req('next')
            print("# Command received: {}".format(step))

            if step == 'train':
                print("# Training", end=' ')
                # TODO: Having a fix number of MB before sync can cause problems
                for i in xrange(nb_batches_before_sync):
                    data = np.asarray(self._worker.recv_mb())
                    print(".", end=' ')
                    self._update_sum(data)
                print("Done")
                import time
                time.sleep(1)
                step = self._worker.send_req(dict(done=nb_batches_before_sync))

                print("Syncing with global params.")
                self._worker.sync_params(synchronous=True)

            if step == 'stop':
                break

        print("All computation done.")
        return self._worker.shared_params[0]  # Return global params


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_port', default=5566, type=int, required=False, help='Port on which the batches will be transfered.')
    parser.add_argument('--control_port', default=5567, type=int, required=False, help='Port on which the control commands will be sent.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    print("Init ...", end=' ')
    bps = BatchedPixelSum(control_port=args.control_port,
                          batch_port=args.batch_port)
    print("Done")

    computed_sum = bps.get_sum()

    # Get actual answer for testing
    with gzip.open("../data/mnist.pkl.gz", 'rb') as f:
        train_set, _, _ = cPickle.load(f)
    real_sum = train_set[0].sum(axis=0, dtype=theano.config.floatX)
    assert_allclose(computed_sum, real_sum)
