from __future__ import absolute_import, print_function, division
import os
import sys

import unittest

import theano
from theano import config
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from platoon.training import global_dynamics as gd


class TestGlobalDynamicsWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.worker = Worker(control_port=5567)
            cls.total_nw = cls.worker.global_size
        except Exception as exc:
            print(exc, file=sys.stderr)
            raise exc

    def setUp(self):
        super(TestOpsWorker, self).setUp()
        SEED = 567
        np.random.seed(SEED)
        self.inp1 = 30 * np.random.random((8, 10, 5)).astype(config.floatX)
        self.sinp1 = theano.shared(self.inp1)
        self.inp2 = 50 * np.random.random((5, 20)).astype(config.floatX)
        self.sinp2 = theano.shared(self.inp2)

    def test_sumSGD_object(self):
        sumsgd = gd.SumSGD()
        sumsgd.make_rule(self.sinp1)
        sumsgd()
        expected = self.inp1 * self.total_nw
        actual = self.sinp1.get_value()
        assert np.allclose(expected, actual)

    def test_sumSGD_list(self):
        sumsgd = gd.SumSGD()
        sumsgd.make_rule([self.sinp1, self.sinp2])
        sumsgd()
        expected = self.inp1 * self.total_nw
        actual = self.sinp1.get_value()
        assert np.allclose(expected, actual)
        expected = self.inp2 * self.total_nw
        actual = self.sinp2.get_value()
        assert np.allclose(expected, actual)

    def test_averageSGD_object(self):
        averagesgd = gd.AverageSGD()
        averagesgd.make_rule(self.sinp1)
        averagesgd()
        expected = self.inp1
        actual = self.sinp1.get_value()
        assert np.allclose(expected, actual)

    def test_averageSGD_list(self):
        averagesgd = gd.AverageSGD()
        averagesgd.make_rule([self.sinp1, self.sinp2])
        averagesgd()
        expected = self.inp1
        actual = self.sinp1.get_value()
        assert np.allclose(expected, actual)
        expected = self.inp2
        actual = self.sinp2.get_value()
        assert np.allclose(expected, actual)

    @classmethod
    def tearDownClass(cls):
        cls.worker.close()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGlobalDynamicsWorker)
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if len(res.failures) != 0 or len(res.errors) != 0:
        sys.exit(1)
