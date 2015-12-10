class ParamSyncRule:
    def update_params(self, worker_params, master_params):
        """ Perform an inplace update of the worker parameters and the central
        parameters based on a certain parameter synchronisation rule.
        """
        raise NotImplementedError()

class EASGD(ParamSyncRule):

    def __init__(self, alpha):
        self.set_alpha(alpha)
        
    def get_alpha(self):
        return self.alpha
        
    def set_alpha(self, alpha):
        self.alpha = alpha
        
    def update_params(self, worker_params, master_params):
        """ Perform an inplace update of the worker parameters and the central
        parameters based on a certain parameter synchronisation rule.
        """
        for p, p_tilde in zip(worker_params, master_params):
            diff = self.alpha * (p - p_tilde)
            p -= diff
            p_tilde += diff