from __future__ import absolute_import
import unittest
from six.moves import reload_module as reload

import numpy as np

from ... import util

try:
    from mpi4py import MPI
    MPI_IMPORTED = True
except:
    MPI_IMPORTED = False


class TestOpToMPI(unittest.TestCase):
    @unittest.skipUnless(MPI_IMPORTED, "Needs mpi4py module")
    def test_op_to_mpi(self):
        reload(util)
        assert util.op_to_mpi('+') == MPI.SUM
        assert util.op_to_mpi("sum") == MPI.SUM
        assert util.op_to_mpi("add") == MPI.SUM
        assert util.op_to_mpi('*') == MPI.PROD
        assert util.op_to_mpi("prod") == MPI.PROD
        assert util.op_to_mpi("product") == MPI.PROD
        assert util.op_to_mpi("mul") == MPI.PROD
        assert util.op_to_mpi("max") == MPI.MAX
        assert util.op_to_mpi("maximum") == MPI.MAX
        assert util.op_to_mpi("min") == MPI.MIN
        assert util.op_to_mpi("minimum") == MPI.MIN

    def test_op_to_mpi_import_fail(self):
        util.MPI = None
        with self.assertRaises(AttributeError):
            util.op_to_mpi('+')

    @unittest.skipUnless(MPI_IMPORTED, "Needs mpi4py module")
    def test_op_to_mpi_op_fail(self):
        reload(util)
        with self.assertRaises(ValueError):
            util.op_to_mpi('asdfasfda')
        with self.assertRaises(ValueError):
            util.op_to_mpi('-')


class TestDtypeToMPI(unittest.TestCase):
    @unittest.skipUnless(MPI_IMPORTED, "Needs mpi4py module")
    def test_dtype_to_mpi(self):
        reload(util)
        assert util.dtype_to_mpi(np.dtype('bool')) == MPI.C_BOOL
        assert util.dtype_to_mpi(np.dtype('int8')) == MPI.INT8_T
        assert util.dtype_to_mpi(np.dtype('uint8')) == MPI.UINT8_T
        assert util.dtype_to_mpi(np.dtype('int16')) == MPI.INT16_T
        assert util.dtype_to_mpi(np.dtype('uint16')) == MPI.UINT16_T
        assert util.dtype_to_mpi(np.dtype('int32')) == MPI.INT32_T
        assert util.dtype_to_mpi(np.dtype('uint32')) == MPI.UINT32_T
        assert util.dtype_to_mpi(np.dtype('int64')) == MPI.INT64_T
        assert util.dtype_to_mpi(np.dtype('uint64')) == MPI.UINT64_T
        assert util.dtype_to_mpi(np.dtype('float32')) == MPI.FLOAT
        assert util.dtype_to_mpi(np.dtype('float64')) == MPI.DOUBLE
        assert util.dtype_to_mpi(np.dtype('complex64')) == MPI.C_FLOAT_COMPLEX
        assert util.dtype_to_mpi(np.dtype('complex128')) == MPI.C_DOUBLE_COMPLEX

    def test_dtype_to_mpi_import_fail(self):
        util.MPI = None
        with self.assertRaises(AttributeError):
            util.dtype_to_mpi('int8')

    @unittest.skipUnless(MPI_IMPORTED, "Needs mpi4py module")
    def test_dtype_to_mpi_dtype_fail(self):
        reload(util)
        with self.assertRaises(TypeError):
            util.dtype_to_mpi('sadfa')
        with self.assertRaises(TypeError):
            util.dtype_to_mpi('')
        # TODO Find how to convert from half type to MPI dtype
        # and use in collectives
        with self.assertRaises(TypeError):
            util.dtype_to_mpi('float16')
