from .base.param_sync_rule import ParamSyncRule


class EASGD(ParamSyncRule):
    """
    Implementation of the EASGD parameter sync rule.

    According to this rule, every N iterations, a worker synchronises his
    parameters with the master parameters. This is done by moving each set of
    parameters toward the other by an amount proportional to the difference
    between the individual params (this proportion is parametrized by `alpha`).

    The sync equations are as follow:
    diff = w_worker - w_master
    w_worker = w_worker - alpha * diff
    w_master = w_master + alpha * diff

    NOTE : if alpha=0 is used, there is no synchronization of the
    parameters meaning that each worker is independently training using SGD.

    This algorithm is described in more details in the following paper:
    http://arxiv.org/abs/1412.6651
    """

    def __init__(self, alpha):
        self.set_alpha(alpha)

    def get_alpha(self):
        return self.alpha

    def set_alpha(self, alpha):
        self.alpha = alpha

    def theano_update(self, local_params):
        # Theano is imported here to avoid a strong dependancy on it.
        import theano
        master_inps = [l.type() for l in local_params]
        master_ups = []
        local_ups = []
        for p_local, p_master in zip(local_params, master_inps):
            diff = self.alpha * (p_local - p_master)
            local_ups.append(p_local - diff)
            master_ups.append(p_master + diff)
        return theano.function(master_inps, master_ups,
                               updates=list(zip(local_params, local_ups)))

    def update_params(self, local_params, master_params):
        for p_local, p_master in zip(local_params, master_params):
            diff = self.alpha * (p_local - p_master)
            p_local -= diff
            p_master += diff


class ASGD(ParamSyncRule):
    def theano_update(self, local_params):
        import theano

        local_vals = [p.get_value(borrow=True, return_internal_type=True)
                      for p in local_params]
        master_inps = [l.type() for l in local_params]
        self.old_locals = [theano.shared(l) for l in local_vals]
        # This updates the global params with the difference between
        # old and current (aka the gradients).
        ret = [m + (p - o) for (m, p, o) in zip(master_inps, local_params,
                                                self.old_locals)]
        # This keeps values before the update for the local params
        ups = list(zip(self.old_locals, ret))
        # This updates the local params to be the same as the global
        ups += list(zip(local_params, ret))
        return theano.function(master_inps, ret, updates=ups)
