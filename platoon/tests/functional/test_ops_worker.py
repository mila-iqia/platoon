from __future__ import absolute_import, print_function, division
import os
import sys

import unittest

import theano
from theano import config
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from platoon import Worker
from platoon import ops


class TestOpsWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.total_nw = int(os.environ['PLATOON_TEST_WORKERS_NUM'])
            cls.worker = Worker(control_port=5567)
        except Exception as exc:
            print(exc, file=sys.stderr)
            raise exc

    def setUp(self):
        super(TestOpsWorker, self).setUp()
        SEED = 567
        np.random.seed(SEED)
        self.inp = 30 * np.random.random((8, 10, 5)).astype(config.floatX)
        self.sinp = theano.shared(self.inp)
        self.out = np.empty_like(self.inp)
        self.sout = theano.shared(self.out)

    def test_all_reduce_sum(self):
        res = ops.AllReduceSum(self.sinp)
        f = theano.function([], [], updates=[(self.sout, res)],
                            profile=True)
        expected = self.total_nw * self.inp
        f()
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)

        # This is faster, because it runs inplace!
        res = ops.AllReduceSum(self.sinp, self.sout)
        f = theano.function([], [], updates=[(self.sout, res)],
                            accept_inplace=True, profile=True)
        expected = self.total_nw * self.inp
        f()
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)

        x = theano.tensor.scalar(dtype=config.floatX)
        res = ops.AllReduceSum(self.sinp, self.sout)
        f = theano.function([x], [], updates=[(self.sout, res / x)],
                            accept_inplace=True, profile=True)
        expected = self.total_nw * self.inp / 2
        f(2)
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)
        expected = self.total_nw * self.inp / 3.14159
        f(3.14159)
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)

        x = theano.tensor.scalar(dtype=config.floatX)
        self.sinp *= x
        res = ops.AllReduceSum(self.sinp, self.sout)
        f = theano.function([x], [], updates=[(self.sout, res)],
                            accept_inplace=True, profile=True)
        expected = self.total_nw * self.inp * 2
        f(2)
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)
        expected = self.total_nw * self.inp * 3.14159
        f(3.14159)
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)

    def test_all_reduce_prod(self):
        res = ops.AllReduceProd(self.sinp)
        f = theano.function([], [], updates=[(self.sout, res)],
                            profile=True)
        expected = self.inp ** self.total_nw
        f()
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)

        # This is faster, because it runs inplace!
        res = ops.AllReduceProd(self.sinp, self.sout)
        f = theano.function([], [], updates=[(self.sout, res)],
                            accept_inplace=True, profile=True)
        expected = self.inp ** self.total_nw
        f()
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)

    def test_all_reduce_maximum(self):
        res = ops.AllReduceMax(self.sinp)
        f = theano.function([], [], updates=[(self.sout, res)],
                            profile=True)
        expected = self.inp
        f()
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)

        # This is faster, because it runs inplace!
        res = ops.AllReduceMax(self.sinp, self.sout)
        f = theano.function([], [], updates=[(self.sout, res)],
                            accept_inplace=True, profile=True)
        expected = self.inp
        f()
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)

    def test_all_reduce_minimum(self):
        res = ops.AllReduceMin(self.sinp)
        f = theano.function([], [], updates=[(self.sout, res)],
                            profile=True)
        expected = self.inp
        f()
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)

        # This is faster, because it runs inplace!
        res = ops.AllReduceMin(self.sinp, self.sout)
        f = theano.function([], [], updates=[(self.sout, res)],
                            accept_inplace=True, profile=True)
        expected = self.inp
        f()
        actual = self.sout.get_value()
        assert np.allclose(expected, actual)

    def test_on_diferent_types(self):
        tmp = np.empty_like(self.inp, dtype='int32')
        stmp = theano.shared(tmp)
        self.assertRaises(TypeError, ops.AllReduceSum, self.sinp, stmp)

    @classmethod
    def tearDownClass(cls):
        cls.worker.close()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOpsWorker)
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if len(res.failures) != 0 or len(res.errors) != 0:
        sys.exit(1)
