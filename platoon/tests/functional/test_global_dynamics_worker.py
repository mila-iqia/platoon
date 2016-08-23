from __future__ import absolute_import, print_function, division
import os
import sys

import unittest

import theano
from theano import config
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from platoon.training import global_dynamics as gd
from platoon.channel.worker import Worker


class TestGlobalDynamicsWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.worker = Worker(control_port=5567)
            cls.total_nw = cls.worker.global_size
            cls.rank = cls.worker.global_rank
        except Exception as exc:
            print(exc, file=sys.stderr)
            raise exc

    def setUp(self):
        super(TestGlobalDynamicsWorker, self).setUp()
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

    def test_EASGD(self):
        lp = np.array([3, 4], dtype=config.floatX)
        if self.rank % 2 != 0:
            lp = -lp
        slp = theano.shared(lp)
        cp = np.array([0, 0], dtype=config.floatX)
        scp = theano.shared(cp)
        alpha = 0.5

        easgd = gd.EASGD()
        easgd.make_rule(slp, scp, alpha)
        easgd()

        if self.total_nw % 2 == 0:
            expectedcp = cp
            actualcp = scp.get_value()
            assert np.allclose(expectedcp, actualcp), (expectedcp, actualcp)
            expectedlp = lp / 2
            actuallp = slp.get_value()
            assert np.allclose(expectedlp, actuallp), (expectedlp, actuallp)
        else:
            expectedcp = lp / 2
            actualcp = scp.get_value()
            assert np.allclose(expectedcp, actualcp), (expectedcp, actualcp)
            expectedlp = lp / 2
            actuallp = slp.get_value()
            assert np.allclose(expectedlp, actuallp), (expectedlp, actuallp)

    def test_Downpour(self):
        lp = np.random.random((2,)).astype(config.floatX)
        slp = theano.shared(lp)
        gp = np.array([0, 1], dtype=config.floatX)
        sgp = theano.shared(gp)
        lau = (self.rank + 1) * np.array([1, 1], dtype=config.floatX)
        slau = theano.shared(lau)

        downpour = gd.Downpour()
        downpour.make_rule(slp, slau, sgp)
        downpour()

        expected = np.array([0, 0], dtype=config.floatX)
        actual = slau.get_value()
        assert np.allclose(expected, actual), (expected, actual)
        expected = sum(np.arange(self.total_nw + 1)) * np.array([1, 1], dtype=config.floatX)
        expected += np.array([0, 1], dtype=config.floatX)
        actual = sgp.get_value()
        assert np.allclose(expected, actual), (expected, actual)
        actual = slp.get_value()
        assert np.allclose(expected, actual), (expected, actual)

    @classmethod
    def tearDownClass(cls):
        cls.worker.close()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGlobalDynamicsWorker)
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if len(res.failures) != 0 or len(res.errors) != 0:
        sys.exit(1)
