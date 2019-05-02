from .optimizer import Optimizer
import pymultinest
import numpy as np
import os

class MultiNest(Optimizer):


    def __init__(self,multi_nest_path,observed=None,model=None):
        super().__init__('Multinest',observed,model)

        # sampling chains directory
        self.nest_path = 'chains/'
        self.nclust_par = -1
        # sampling efficiency (parameter, ...)
        self.sampling_eff = 'parameter'
        # number of live points
        self.n_live_points = 1000
        # maximum no. of iterations (0=inf)
        self.max_iter = 0
        # search for multiple modes
        self.multimodes = True
        #parameters on which to cluster, e.g. if nclust_par = 3, it will cluster on the first 3 parameters only.
        #if ncluster_par = -1 it clusters on all parameters
        self.nclust_par = -1
        # maximum number of modes
        self.max_modes = 100
        # run in constant efficiency mode
        self.const_eff = False
        # set log likelihood tolerance. If change is smaller, multinest will have converged
        self.evidence_tolerance = 0.5
        self.mode_tolerance = -1e90
        # importance nested sampling
        self.imp_sampling = False  

        self.dir_multinest = multi_nest_path  

    def compute_fit(self):

        data = self._observed.spectrum
        datastd = self._observed.errorBar

        def multinest_loglike(cube, ndim, nparams):
            # log-likelihood function called by multinest
            fit_params_container = np.asarray([cube[i] for i in range(len(self.fitting_parameters))])
            chi_t = self.chisq_trans(fit_params_container, data, datastd)
            loglike = (-1.)*np.sum(np.log(datastd*np.sqrt(2*np.pi))) - 0.5 * chi_t
            return loglike

        def multinest_uniform_prior(cube, ndim, nparams):
            # prior distributions called by multinest. Implements a uniform prior
            # converting parameters from normalised grid to uniform prior
            for idx,bounds in enumerate(self.fit_boundaries):
                bound_min,bound_max = bounds
                cube[idx] = (cube[idx] * (bound_max-bound_min)) + bound_min
   
        datastd_mean = np.mean(datastd)
        ndim = len(self.fitting_parameters)
        self.info('Beginning fit......')
        pymultinest.run(LogLikelihood=multinest_loglike,
                        Prior=multinest_uniform_prior,
                        n_dims=ndim,
                        multimodal=self.multimodes,
                        n_clustering_params=self.nclust_par,
                        max_modes=self.max_modes,
                        outputfiles_basename=os.path.join(self.dir_multinest, '1-'),
                        const_efficiency_mode = self.const_eff,
                        importance_nested_sampling = self.imp_sampling,
                        resume = False,
                        verbose = True,
                        sampling_efficiency = self.sampling_eff,
                        evidence_tolerance = self.evidence_tolerance,
                        mode_tolerance = self.mode_tolerance,
                        n_live_points = self.n_live_points,
                        max_iter= self.max_iter,
                        init_MPI=False)
        
        self.info('Fit complete.....')