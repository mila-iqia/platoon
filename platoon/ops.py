# -*- coding: utf-8 -*-
"""
:mod:`ops` -- Theano Ops for Worker interface
=============================================

.. module:: ops
   :platform: Unix
   :synopsis: Contains AllReduce Theano Op and builder function for each
              reduce operation type.

"""
from __future__ import absolute_import, print_function
import sys

from six.moves import xrange

try:
    import theano
    from theano.gradient import grad_not_implemented
    from theano.gpuarray.basic_ops import as_gpuarray_variable
except ImportError as exc:
    print("ERROR! On {}:".format(__name__), exc, file=sys.stderr)
    theano = None

from .channel.worker import Worker


if theano:
    class AllReduce(theano.Op):
        """Wrapper of :class:`channel.worker.Worker`.

        For full documentation, see builder functions:
           * :func:`AllReduceSum`
           * :func:`AllReduceProd`
           * :func:`AllReduceMax`
           * :func:`AllReduceMin`

        :param scalar_op: Representation of collective reduce operation type.
        :type scalar_op: {str, :ref:`theano.scalar.add`, :ref:`theano.scalar.mul`,
                          :ref:`theano.scalar.maximum`, :ref:`theano.scalar.minimum`}

        .. seealso:: module :mod:`channel.worker`

        .. versionadded:: 0.6.0

        """
        __props__ = ("scalar_op", )

        def __init__(self, scalar_op, inplace=False, worker=None):
            if worker is not None:
                if isinstance(worker, Worker):
                    self.worker = worker
                else:
                    raise TypeError("Argument `worker` is not of platoon.Worker type.")
            else:
                try:
                    self.worker = Worker()  # Get singleton instance
                except TypeError:
                    raise AttributeError("Worker instance has not been created yet.")
            # This is because I have not found a way to use half-types through MPI
            self._f16_ok = not self.worker._multinode
            self.scalar_op = scalar_op
            self.inplace = inplace

        def __str__(self):
            if self.inplace:
                return "AllReduce{%s,inplace}<gpuarray.collectives>" % (str(self.scalar_op).capitalize())
            else:
                return "AllReduce{%s,no_inplace}<gpuarray.collectives>" % (str(self.scalar_op).capitalize())

        def make_node(self, src, dest=None):
            if dest is None:
                inputs = [src]
                if self.inplace:
                    self.inplace_pattern = {0: 0}
                else:
                    self.inplace_pattern = {}
            else:
                inputs = [src, dest]
                self.inplace = True
                self.inplace_pattern = {0: 1}
            self.destroy_map = dict((o, [i]) for o, i in self.inplace_pattern.items())
            inputs = [as_gpuarray_variable(i, self.worker.ctx_name) for i in inputs]
            if dest is not None:
                if not inputs[0].type == inputs[1].type:
                    raise TypeError("`src` and `dest` must have the same Type:",
                                    (inputs[0].type, inputs[1].type))
            out_type = inputs[0].type.clone()
            return theano.Apply(self, inputs, [out_type()])

        def infer_shapes(self, node, shapes):
            return [shapes[0]]

        def perform(self, node, inputs, outputs):
            out = outputs[0]
            src = inputs[0]
            if len(node.inputs) == 2:  # If inplace op
                dest = inputs[1]
                self.worker.all_reduce(src, str(self.scalar_op), dest)
                out[0] = dest
            elif self.inplace:
                self.worker.all_reduce(src, str(self.scalar_op), src)
                out[0] = src
            else:
                out[0] = self.worker.all_reduce(src, str(self.scalar_op))

        def grad(self, inputs, ograds):
            return [grad_not_implemented(self, i, inputs[i]) for i in xrange(len(inputs))]

    def AllReduceSum(src, dest=None, inplace=False, worker=None):
        """
        Element-wise sum  of `src` GPU tensor across all
        Platoon worker processes.

        Parameters
        ----------
        src : GPU tensor (array-like)
           Input array.
        dest : GPU tensor (array-like), optional
           Output array. If None (default) is given, then an GPU array-like
           will be returned with result, which has the same shape and datatype
           as `src`.
        inplace : bool, optional
           If True, then operation will happen inplace and the result will be
           written in array `src`.
        worker : :class:`channel.worker.Worker`, optional
           Platoon Worker instance unique to a single process which will be used
           to execute the operation. If None (default) is given, the singleton
           instance will be used.

        Returns
        -------
        result : GPU tensor (array-like)
           Result array will be `dest` if it was specified in the arguments,
           `src` if `inplace` is True, else a new variable which points to
           operation's result.

        Notes
        -----
        * If `dest` is given, then the Op is inplace in Theano sense.
        * If a `worker` is not given, then a Worker instance must have been
          already instantiated.

        Raises
        ------
        TypeError
           If `worker` specified is not of type :class:`channel.worker.Worker`
           or if `src` and `dest` are not of the same Theano Type.
        AttributeError
           If singleton Worker has not been instantiated yet.

        .. versionadded:: 0.6.0

        """
        return AllReduce(theano.scalar.add, inplace, worker)(src, dest)

    def AllReduceProd(src, dest=None, inplace=False, worker=None):
        """
        Element-wise multiplication of `src` GPU tensor across all
        Platoon worker processes.

        .. seealso::
           Function :func:`AllReduceSum`
              For documentation on parameters, return variables, notes and
              raises.

        .. versionadded:: 0.6.0

        """
        return AllReduce(theano.scalar.mul, inplace, worker)(src, dest)

    def AllReduceMax(src, dest=None, inplace=False, worker=None):
        """
        Find element-wise maximum of `src` GPU tensor across all
        Platoon worker processes.

        .. seealso::
           Function :func:`AllReduceSum`
              For documentation on parameters, return variables, notes and
              raises.

        .. versionadded:: 0.6.0

        """
        return AllReduce(theano.scalar.maximum, inplace, worker)(src, dest)

    def AllReduceMin(src, dest=None, inplace=False, worker=None):
        """
        Find element-wise minimum of `src` GPU tensor across all
        Platoon worker processes.

        .. seealso::
           Function :func:`AllReduceSum`
              For documentation on parameters, return variables, notes and
              raises.

        .. versionadded:: 0.6.0

        """
        return AllReduce(theano.scalar.minimum, inplace, worker)(src, dest)
else:
    AllReduce = None
    AllReduceSum = None
    AllReduceProd = None
    AllReduceMax = None
    AllReduceMin = None
