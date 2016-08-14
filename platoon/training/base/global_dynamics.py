from abc import ABCMeta

import six

from ...worker import Worker


@six.add_metaclass(ABCMeta)
class GlobalDynamics(object):
    """Abstract class which declares the methods and properties that need to
    be implemented by a synchronous global dynamics rule.

    """
    def __init__(self, shared_inputs, worker=None):
        if worker is not None:
            self.worker = worker
        self.__call__ = self.make_rule(shared_inputs)

    def __call__(self):
        pass

    def make_rule(self, shared_inputs):
        raise NotImplementedError

    @property
    def worker(self):
        """Worker class instance used for global operations"""
        return self._worker

    @worker.setter
    def worker(self, inst):
        if not isinstance(inst, Worker):
            raise TypeError("`inst` argument is not a Worker")
        self._worker = inst
