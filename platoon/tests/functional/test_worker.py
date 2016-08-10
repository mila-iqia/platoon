from __future__ import absolute_import, print_function
import os
import sys

import unittest

import theano
import numpy as np

from platoon.worker import Worker


class TestWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.total_nw = int(os.environ['PLATOON_TEST_WORKERS_NUM'])
            cls.worker = Worker(control_port=5567)
        except Exception as exc:
            print(exc, file=sys.stderr)
            raise exc

    def test_global_size(self):
        assert self.worker.global_size == self.total_nw

    def test_interface1(self):
        try:
            inp = np.arange(32, dtype='float64')
            sinp = theano.shared(inp)
            out = np.empty_like(inp)
            sout = theano.shared(out)
            self.worker.all_reduce(sinp, '+', sout)
            expected = self.total_nw * inp
            actual = sout.get_value()
            assert np.allclose(expected, actual)
        except Exception as exc:
            print(exc, file=sys.stderr)
            raise exc

    def test_interface2(self):
        try:
            inp = np.arange(32, dtype='float64')
            sinp = theano.shared(inp)
            self.worker.all_reduce(sinp, '+', sinp)
            expected = self.total_nw * inp
            actual = sinp.get_value()
            assert np.allclose(expected, actual)
        except Exception as exc:
            print(exc, file=sys.stderr)
            raise exc

    def test_interface3(self):
        try:
            inp = np.arange(32, dtype='float64')
            sinp = theano.shared(inp)
            sout = self.worker.all_reduce(sinp, '+')
            expected = self.total_nw * inp
            actual = sout.get_value()
            assert np.allclose(expected, actual)
        except Exception as exc:
            print(exc, file=sys.stderr)
            raise exc

    def test_linked_shared(self):
        try:
            inp = np.arange(32, dtype='float64')
            sinp = theano.shared(inp)
            out = np.empty_like(inp)
            sout = theano.shared(out)

            if self.worker._multinode:
                try:
                    self.worker.shared_arrays[sout]
                    self.fail("'sout' has not been linked yet to a shared buffer")
                except KeyError:
                    pass
                try:
                    self.worker.shared_arrays[sinp]
                    self.fail("'sinp' has not been linked yet to a shared buffer")
                except KeyError:
                    pass

            self.worker.all_reduce(sinp, '+', sout)

            if self.worker._multinode:
                try:
                    self.worker.shared_arrays[sout]
                except KeyError:
                    self.fail("sout should have been linked to a shared buffer")
                try:
                    self.worker.shared_arrays[sinp]
                    self.fail("'sinp' has not been linked yet to a shared buffer")
                except KeyError:
                    pass

            expected = self.total_nw * inp
            actual = sout.get_value()
            assert np.allclose(expected, actual)

            self.worker.all_reduce(sout, '*', sout)

            if self.worker._multinode:
                try:
                    self.worker.shared_arrays[sout]
                except KeyError:
                    self.fail("sout should have been linked to a shared buffer")
                try:
                    self.worker.shared_arrays[sinp]
                    self.fail("'sinp' has not been linked yet to a shared buffer")
                except KeyError:
                    pass

            expected = expected ** self.total_nw
            actual = sout.get_value()
            assert np.allclose(expected, actual)
        except Exception as exc:
            print(exc, file=sys.stderr)
            raise exc

    @classmethod
    def tearDownClass(cls):
        cls.worker.close()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWorker)
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if len(res.failures) != 0 or len(res.errors) != 0:
        sys.exit(1)
