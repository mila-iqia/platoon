class ParamSyncRule(object):
    """
    Abstract parameter synchronisation rule.

    This abstract class defines the interface that should be followed by
    implementations of parameter synchronization rules for distributed
    training.
    """

    def make_update_function(self, local_params):
        """Return a function that will be called with the current value of the
        master parameters and should update them inplace.  This
        function must also update the values of local_params (that are
        shared values) as a side effect.
        """
        try:
            f = self.theano_update(local_params)

            def update(master_params, f=f):
                new_master_values = f(*master_params)
                for p, v in zip(master_params, new_master_values):
                    p[:] = v
        except NotImplementedError:
            def update(master_params, local_params=local_params,
                       update_params=self.update_params):
                local_param_values = [p.get_value() for p in local_params]
                update_params(local_param_values, master_params)
                for p, v in zip(local_params, local_param_values):
                    p.set_value(v)
        return update

    def theano_update(self, local_params):
        """Compile and return a theano function that will update the local
        params and return new values for the master params.

        This function is preferred to update_params below.
        """
        raise NotImplementedError()

    def update_params(self, local_params, master_params):
        """Perform an inplace update of the local and master params according
        to some update rule.

        This function need not be implemented if theano_update is
        overridden.

        """
        raise NotImplementedError()
