from .optimizer import Optimizer
import nestle
import numpy as np
import os
import time
class NestleOptimizer(Optimizer):

    def __init__(self,observed=None,model=None):
        super().__init__('Nestle',observed,model)
        self._nlive = 1500     # number of live points
        self._method = 'multi' # use MutliNest algorithm
        
        self._tol = 0.5        # the stopping criterion
    

    @property
    def tolerance(self):
        return self._tol
    
    @tolerance.setter
    def tolerance(self,value):
        self._tol = value
    

    @property
    def numLivePoints(self):
        return self._nlive
    
    @numLivePoints.setter
    def numLivePoints(self,value):
        self._nlive = value



    def compute_fit(self):

        data = self._observed.spectrum
        datastd = self._observed.errorBar
        sqrtpi = np.sqrt(2*np.pi)
        def nestle_loglike(params):
            # log-likelihood function called by multinest
            fit_params_container = np.array(params)
            chi_t = self.chisq_trans(fit_params_container, data, datastd)
            loglike = -np.sum(np.log(datastd*sqrtpi)) - 0.5 * chi_t
            return loglike

        def nestle_uniform_prior(theta):
            # prior distributions called by multinest. Implements a uniform prior
            # converting parameters from normalised grid to uniform prior
            cube = []

            for idx,bounds in enumerate(self.fit_boundaries):
                bound_min,bound_max = bounds
                cube.append((theta[idx] * (bound_max-bound_min)) + bound_min)
            
            return tuple(cube)
            

        ndim = len(self.fitting_parameters)
        self.info('Beginning fit......')
        ndims = ndim        # two parameters

        t0 = time.time()
        
        res = nestle.sample(nestle_loglike, nestle_uniform_prior, ndims, method='multi', npoints=self.numLivePoints, dlogz=self.tolerance,callback=nestle.print_progress)
        t1 = time.time()

        timenestle = (t1-t0)

        print("Time taken to run 'Nestle' is {} seconds".format(timenestle))
        
        self.info('Fit complete.....')

        print(res)