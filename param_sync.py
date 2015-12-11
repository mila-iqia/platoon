class ParamSyncRule:
    def update_params(self, local_params, master_params):
        """
        Perform an inplace update of the master parameters and returns the 
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
        
    def update_params(self, local_params, master_params):
        """
        Perform an inplace update of the worker parameters and the central
        parameters based on a certain parameter synchronisation rule.
        """
        for p_local, p_master in zip(local_params, master_params):
            diff = self.alpha * (p_local - p_master)
            
            # The -= and += operations are inplace for nupy arrays
            p_local -= diff
            p_master += diff
