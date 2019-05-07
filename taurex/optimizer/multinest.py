from .optimizer import Optimizer
import pymultinest
import numpy as np
import os

class MultiNestOptimizer(Optimizer):


    def __init__(self,multi_nest_path=None,observed=None,model=None,
                sampling_efficiency='parameter',
                num_live_points=1500,
                max_iterations=0,
                search_multi_modes = True,
                num_params_cluster=-1,
                maximum_modes=100,
                constant_efficiency_mode=False,
                evidence_tolerance=0.5,
                mode_tolerance=-1e90,
                importance_sampling=True,
                resume=False,
                verbose_output=True):
        super().__init__('Multinest',observed,model)

        # sampling chains directory
        self.nest_path = 'chains/'
        self.nclust_par = -1
        # sampling efficiency (parameter, ...)
        self.sampling_eff = sampling_efficiency
        # number of live points
        self.n_live_points = num_live_points
        # maximum no. of iterations (0=inf)
        self.max_iter = max_iteration
        # search for multiple modes
        self.multimodes = search_multi_modes
        #parameters on which to cluster, e.g. if nclust_par = 3, it will cluster on the first 3 parameters only.
        #if ncluster_par = -1 it clusters on all parameters
        self.nclust_par = num_params_cluster
        # maximum number of modes
        self.max_modes = maximum_modes
        # run in constant efficiency mode
        self.const_eff = constant_efficiency_mode
        # set log likelihood tolerance. If change is smaller, multinest will have converged
        self.evidence_tolerance = evidence_tolerance
        self.mode_tolerance = mode_tolerance
        # importance nested sampling
        self.imp_sampling = importance_sampling

        self.dir_multinest = multi_nest_path  

        self.resume = resume
        self.verbose = verbose_output


    def compute_fit(self):

        data = self._observed.spectrum
        datastd = self._observed.errorBar
        sqrtpi = np.sqrt(2*np.pi)
        def multinest_loglike(cube, ndim, nparams):
            # log-likelihood function called by multinest
            fit_params_container = np.array([cube[i] for i in range(len(self.fitting_parameters))])
            chi_t = self.chisq_trans(fit_params_container, data, datastd)
            
            #print('---------START---------')
            #print('chi_t',chi_t)
            #print('LOG',loglike)
            loglike = -np.sum(np.log(datastd*sqrtpi)) - 0.5 * chi_t
            return loglike

        def multinest_uniform_prior(cube, ndim, nparams):
            # prior distributions called by multinest. Implements a uniform prior
            # converting parameters from normalised grid to uniform prior
            #print(type(cube))
            for idx,bounds in enumerate(self.fit_boundaries):
                # print(idx,self.fitting_parameters[idx])
                bound_min,bound_max = bounds
                cube[idx] = (cube[idx] * (bound_max-bound_min)) + bound_min
                #print('CUBE idx',cube[idx])
            #print('-----------')
        status = None
        def dump_call(nSamples,nlive,nPar,physLive,posterior,paramConstr,maxloglike,logZ,INSlogZ,logZerr,context):
            status = (nSamples,nlive,nPar,physLive,posterior,paramConstr,maxloglike,logZ,INSlogZ,logZerr,context)


        datastd_mean = np.mean(datastd)
        ndim = len(self.fitting_parameters)
        self.info('Beginning fit......')
        pymultinest.run(LogLikelihood=multinest_loglike,
                        Prior=multinest_uniform_prior,
                        n_dims=ndim,
                        multimodal=self.multimodes,
                        n_clustering_params=self.nclust_par,
                        max_modes=self.max_modes,
                        outputfiles_basename=os.path.join(self.dir_multinest, '3-'),
                        const_efficiency_mode = self.const_eff,
                        importance_nested_sampling = self.imp_sampling,
                        resume = self.resume,
                        verbose = self.verbose,
                        sampling_efficiency = self.sampling_eff,
                        evidence_tolerance = self.evidence_tolerance,
                        mode_tolerance = self.mode_tolerance,
                        n_live_points = self.n_live_points,
                        max_iter= self.max_iter
                        )
        
        self.info('Fit complete.....')

        rank = 0

        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        except:
            pass

        #Only allow rank 0 to do any further processing
        if rank == 0:
            return status
        else:
            return None

    
    def process_multinest_status(self,status):
        pass
