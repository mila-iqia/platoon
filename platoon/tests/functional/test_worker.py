from __future__ import absolute_import, print_function
import os
import sys

import unittest

from pygpu import gpuarray
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from platoon import Worker


class TestWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.total_nw = int(os.environ['PLATOON_TEST_WORKERS_NUM'])
            cls.worker = Worker(control_port=5567)
            cls.ctx = cls.worker.gpuctx
        except Exception as exc:
            print(exc, file=sys.stderr)
            raise exc

    def test_is_singleton(self):
        inst = Worker()
        assert inst is self.worker
        print("The following warning is produced by testing procedure:", file=sys.stderr)
        inst = Worker(123413)
        assert inst is self.worker

    def test_global_size(self):
        assert self.worker.global_size == self.total_nw

    def test_interface1(self):
        inp = np.arange(32, dtype='float64')
        sinp = gpuarray.asarray(inp, context=self.ctx)
        out = np.empty_like(inp)
        sout = gpuarray.asarray(out, context=self.ctx)
        self.worker.all_reduce(sinp, '+', sout)
        expected = self.total_nw * inp
        actual = np.asarray(sout)
        assert np.allclose(expected, actual)

    def test_interface2(self):
        inp = np.arange(32, dtype='float64')
        sinp = gpuarray.asarray(inp, context=self.ctx)
        self.worker.all_reduce(sinp, '+', sinp)
        expected = self.total_nw * inp
        actual = np.asarray(sinp)
        assert np.allclose(expected, actual)

    def test_interface3(self):
        inp = np.arange(32, dtype='float64')
        sinp = gpuarray.asarray(inp, context=self.ctx)
        sout = self.worker.all_reduce(sinp, '+')
        expected = self.total_nw * inp
        actual = np.asarray(sout)
        assert np.allclose(expected, actual)

    def test_linked_shared(self):
        inp = np.arange(32, dtype='float64')
        sinp = gpuarray.asarray(inp, context=self.ctx)
        insize = sinp.size * sinp.itemsize
        out = np.empty_like(inp)
        sout = gpuarray.asarray(out, context=self.ctx)
        outsize = sout.size * sout.itemsize

        if self.worker._multinode:
            try:
                self.worker.shared_arrays[outsize]
                self.fail("'sout''s size has not been linked yet to a shared buffer")
            except KeyError:
                pass
            try:
                self.worker.shared_arrays[insize]
                self.fail("'sinp''s size has not been linked yet to a shared buffer")
            except KeyError:
                pass

        self.worker.all_reduce(sinp, '+', sout)

        if self.worker._multinode:
            try:
                self.worker.shared_arrays[outsize]
            except KeyError:
                self.fail("`sout`'s size should have been linked to a shared buffer")
            try:
                self.worker.shared_arrays[insize]
            except KeyError:
                self.fail("`sinp`'s size should have been linked to a shared buffer")

        expected = self.total_nw * inp
        actual = np.asarray(sout)
        assert np.allclose(expected, actual)

        self.worker.all_reduce(sout, '*', sout)

        if self.worker._multinode:
            try:
                self.worker.shared_arrays[outsize]
            except KeyError:
                self.fail("`sout`'s size should have been linked to a shared buffer")
            try:
                self.worker.shared_arrays[insize]
            except KeyError:
                self.fail("`sinp`'s size should have been linked to a shared buffer")

        expected = expected ** self.total_nw
        actual = np.asarray(sout)
        assert np.allclose(expected, actual)

    @classmethod
    def tearDownClass(cls):
        cls.worker.close()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWorker)
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if len(res.failures) != 0 or len(res.errors) != 0:
        sys.exit(1)
