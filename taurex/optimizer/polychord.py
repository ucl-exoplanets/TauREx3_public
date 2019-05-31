from .optimizer import Optimizer
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
import numpy as np
import os

from taurex.util.util import read_table,read_error_line,read_error_into_dict,quantile_corner,recursively_save_dict_contents_to_output

class PolyChordOptimizer(Optimizer):


    def __init__(self,polychord_path=None,observed=None,model=None,
                num_live_points=1500,
                max_iterations=0,
                maximum_modes=100,
                cluster=True,
                evidence_tolerance=0.5,
                mode_tolerance=-1e90,
                resume=False,
                verbosity=1):
        super().__init__('Multinest',observed,model)

        # number of live points
        self.n_live_points = int(num_live_points)
        # maximum no. of iterations (0=inf)
        self.max_iter = int(max_iterations)
        # search for multiple modes
        #parameters on which to cluster, e.g. if nclust_par = 3, it will cluster on the first 3 parameters only.
        #if ncluster_par = -1 it clusters on all parameters
        # maximum number of modes
        self.max_modes = int(maximum_modes)
        self.do_clustering = cluster
        # run in constant efficiency mode
        # set log likelihood tolerance. If change is smaller, multinest will have converged
        self.evidence_tolerance = evidence_tolerance
        self.mode_tolerance = mode_tolerance
        # importance nested sampling

        self.dir_polychord = polychord_path

        self.resume = resume
        self.verbose = verbosity


    def compute_fit(self):

        data = self._observed.spectrum
        datastd = self._observed.errorBar
        sqrtpi = np.sqrt(2*np.pi)

        ndim = len(self.fitting_parameters)
        def polychord_loglike(cube):
            # log-likelihood function called by polychord
            fit_params_container = np.array([cube[i] for i in range(len(self.fitting_parameters))])
            chi_t = self.chisq_trans(fit_params_container, data, datastd)
            
            #print('---------START---------')
            #print('chi_t',chi_t)
            #print('LOG',loglike)
            loglike = -np.sum(np.log(datastd*sqrtpi)) - 0.5 * chi_t
            return loglike,[0.0]

        def polychord_uniform_prior(hypercube):
            # prior distributions called by polychord. Implements a uniform prior
            # converting parameters from normalised grid to uniform prior
            #print(type(cube))
            cube = [0.0]*ndim

            for idx,bounds in enumerate(self.fit_boundaries):
                # print(idx,self.fitting_parameters[idx])
                bound_min,bound_max = bounds
                cube[idx] = (hypercube[idx] * (bound_max-bound_min)) + bound_min
                #print('CUBE idx',cube[idx])
            #print('-----------')
            return cube
        status = None

        


        datastd_mean = np.mean(datastd)
        

        settings = PolyChordSettings(ndim,1)
        settings.nlive     = ndim * 25
        settings.num_repeats = ndim * 5
        settings.do_clustering = True
        settings.num_repeats=ndim
        settings.precision_criterion = self.evidence_tolerance
        settings.logzero = -1e70
        settings.read_resume = self.resume

        self.warning('Number of dimensions {}'.format(ndim))
        self.warning('Fitting parameters {}'.format(self.fitting_parameters))

        self.info('Beginning fit......')
        output = pypolychord.run_polychord(polychord_loglike, ndim, 1, settings, polychord_uniform_prior)
    
        print(output)

