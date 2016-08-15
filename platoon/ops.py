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
from .util import PlatoonError


if theano:
    class AllReduce(theano.Op):
        __props__ = ("scalar_op", )

        def __init__(self, scalar_op, worker=None):
            if worker is not None:
                if isinstance(worker, Worker):
                    self.worker = worker
                else:
                    raise TypeError("Argument `worker` is not of platoon.Worker type.")
            else:
                try:
                    self.worker = Worker()  # Get singleton instance
                except TypeError:
                    raise PlatoonError("A worker instance has not been created yet.")
            # This is because I have not found a way to use half-types through MPI
            self._f16_ok = not self.worker._multinode
            self.scalar_op = scalar_op

        def __str__(self):
            if self.inplace_pattern:
                return "AllReduce{%s,inplace}<gpuarray.collectives>" % (str(self.scalar_op).capitalize())
            else:
                return "AllReduce{%s,no_inplace}<gpuarray.collectives>" % (str(self.scalar_op).capitalize())

        def make_node(self, src, dest=None):
            if dest is None:
                inputs = [src]
                self.inplace_pattern = {}
            else:
                inputs = [src, dest]
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
            else:
                out[0] = self.worker.all_reduce(src, str(self.scalar_op))

        def grad(self, inputs, ograds):
            return [grad_not_implemented(self, i, inputs[i]) for i in xrange(len(inputs))]

    def AllReduceSum(src, dest=None, worker=None):
        return AllReduce(theano.scalar.add, worker)(src, dest)

    def AllReduceProd(src, dest=None, worker=None):
        return AllReduce(theano.scalar.mul, worker)(src, dest)

    def AllReduceMax(src, dest=None, worker=None):
        return AllReduce(theano.scalar.maximum, worker)(src, dest)

    def AllReduceMin(src, dest=None, worker=None):
        return AllReduce(theano.scalar.minimum, worker)(src, dest)
else:
    AllReduce = None
    AllReduceSum = None
    AllReduceProd = None
    AllReduceMax = None
    AllReduceMin = None
