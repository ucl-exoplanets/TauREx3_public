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
        self.n_live_points = int(num_live_points)
        # maximum no. of iterations (0=inf)
        self.max_iter = int(max_iterations)
        # search for multiple modes
        self.multimodes = int(search_multi_modes)
        #parameters on which to cluster, e.g. if nclust_par = 3, it will cluster on the first 3 parameters only.
        #if ncluster_par = -1 it clusters on all parameters
        self.nclust_par = int(num_params_cluster)
        # maximum number of modes
        self.max_modes = int(maximum_modes)
        # run in constant efficiency mode
        self.const_eff = constant_efficiency_mode
        # set log likelihood tolerance. If change is smaller, multinest will have converged
        self.evidence_tolerance = evidence_tolerance
        self.mode_tolerance = mode_tolerance
        # importance nested sampling
        self.imp_sampling = importance_sampling

        self.dir_multinest = os.path.join(multi_nest_path, '1-') 

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
                        outputfiles_basename=self.dir_multinest,
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

    
    def process_multinest_status(self,status):
        pass


    def write_optimizer(self,output):
        opt = super().write_optimizer(output)

        # sampling efficiency (parameter, ...)
        opt.write_scalar('sampling_eff ',self.sampling_eff)
        # number of live points
        opt.write_scalar('num_live_points',self.n_live_points)
        # maximum no. of iterations (0=inf)
        opt.write_scalar('max_iterations',self.max_iter)
        # search for multiple modes
        opt.write_scalar('search_multimodes',self.multimodes)
        #parameters on which to cluster, e.g. if nclust_par = 3, it will cluster on the first 3 parameters only.
        #if ncluster_par = -1 it clusters on all parameters
        opt.write_scalar('nclust_parameter',self.nclust_par)
        # maximum number of modes
        opt.write_scalar('max_modes',self.max_modes)
        # run in constant efficiency mode
        opt.write_scalar('constant_effeciency',self.const_eff)
        # set log likelihood tolerance. If change is smaller, multinest will have converged
        opt.write_scalar('evidence_tolerance',self.evidence_tolerance)
        opt.write_scalar('mode_tolerance',self.mode_tolerance)
        # importance nested sampling
        opt.write_scalar('importance_sampling ',self.imp_sampling)


        return opt
    
    def write_fit(self,output):
        fit = super().write_fit(output)

        return fit