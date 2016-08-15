from __future__ import absolute_import, print_function, division

from ...util import PlatoonError
from ...worker import Worker
from ...ops import AllReduceSum


class GlobalDynamics(object):
    """Abstract class which declares the methods and properties that need to
    be implemented by a synchronous global dynamics rule.

    Parameters
    ----------
    worker : :ref:`platoon.channel.Worker`, optional
        A reference to Worker's instance (which is currently singleton)

    """
    def __init__(self, worker=None):
        self._worker = None
        if worker is not None:
            self.worker = worker
        self._fn = None

    def __call__(self):
        if self._fn is None:
            raise NotImplementedError("Functionality has not been specified.\n
                                      Please use {} method to setup GlobalDynamics
                                      for a set of Variables\nor supply your own
                                      using {} property.".format(
                                          repr(self.make_rule), repr(self.fn)))
        self._fn()

    @property
    def worker(self):
        """Worker class instance used for global operations"""
        if self._worker is None:
            try:
                self._worker = Worker()  # Draw singleton instance
            except TypeError:
                raise AttributeError("Worker instance has not been created yet.")
        return self._worker

    @worker.setter
    def worker(self, inst):
        if not isinstance(inst, Worker):
            raise TypeError("Argument `inst` is not of platoon.Worker type.")
        self._worker = inst

    @property
    def fn(self):
        """Internal function implementing global dynamics. Does not accept
        parameters. Global optimization must be done though shared variables.

        Supplying your own internal function is your responsibility. It must be
        able to be called like this: `fn()`.

        """
        return self._fn

    @property.setter
    def fn(self, inst):
        self._fn = inst

    def make_rule(self, *args):
        """Create GlobalDynamics optimization function for local data in `args`.

        Implementation in a child class must return a callable object which
        expects no arguments. User must be careful to create a function which
        uses shared objects in order to update local model parameters, such as
        Theano Shared Variables.

        """
        raise NotImplementedError(self.make_rule.__doc__)


class _GlobalDynamicsNoSet(GlobalDynamics):
    @property.setter
    def fn(self, inst):
        raise AttributeError("Cannot set internal function. Use {} method.".format(
            repr(self.make_rule)))


class SGD(_GlobalDynamicsNoSet):
    """Synchronous Stochastic Gradient Descent:

    It sums or averages model parameter updates found separately (and
    concurrently) by workers which are training on (different) random
    mini-batches of a dataset.

    Parameters
    ----------
    average : bool, optional
        If True, it will normalize the summation of model param updates across
        all workers with the number of workers participating in optimization.
    worker : :ref:`platoon.channel.Worker`
        See :class:`GlobalDynamics`.

    """
    def __init__(self, average=False, worker=None):
        self.average = average
        super(SGD, self).__init__(worker)

    def make_rule(self, local_updates):
        """Makes global SGD rule for the parameters in `args`.

        Parameters
        ----------
        local_updates: {theano.SharedVariable, list of theano.SharedVariable}
            These variables represent the updates found
            by local optimization dynamics on the model's parameters.

        """
        import theano
        if isinstance(local_updates, theano.SharedVariable):
            local_updates = [local_updates]
        global_updates = []
        for update in local_updates:
            gup = AllReduceSum(update, update)
            if self.average:
                gup /= self.worker.global_size
            global_updates.append(gup)
        self._fn = theano.function([], [],
                                   updates=list(zip(local_updates, global_updates)),
                                   accept_inplace=True)


class SumSGD(SGD):
    """Synchronous Stochastic Gradient Descent: summing version

    See Also
    --------
    :ref:`SGD`

    """
    def __init__(self, worker=None):
        super(SumSGD, self).__init__(average=False, worker=worker)


class AverageSGD(SGD):
    """Synchronous Stochastic Gradient Descent: averaging version

    See Also
    --------
    :ref:`SGD`

    """
    def __init__(self, worker=None):
        super(AverageSGD, self).__init__(average=True, worker=worker)
