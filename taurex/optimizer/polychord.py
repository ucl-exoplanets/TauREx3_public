from .optimizer import Optimizer
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
import numpy as np
import os

from taurex.util.util import read_table, read_error_line, read_error_into_dict, quantile_corner, recursively_save_dict_contents_to_output, weighted_avg_and_std


class PolyChordOptimizer(Optimizer):

    def __init__(self, polychord_path=None, observed=None, model=None,
                 num_live_points=1500,
                 max_iterations=0,
                 maximum_modes=100,
                 cluster=True,
                 evidence_tolerance=0.5,
                 mode_tolerance=-1e90,
                 resume=False,
                 verbosity=1, sigma_fraction=0.1):
        super().__init__('Multinest', observed, model, sigma_fraction)

        # number of live points
        self.n_live_points = int(num_live_points)
        # maximum no. of iterations (0=inf)
        self.max_iter = int(max_iterations)
        # search for multiple modes
        # parameters on which to cluster, e.g. if nclust_par = 3, it will cluster on the first 3 parameters only.
        # if ncluster_par = -1 it clusters on all parameters
        # maximum number of modes
        self.max_modes = int(maximum_modes)
        self.do_clustering = cluster
        # run in constant efficiency mode
        # set log likelihood tolerance. If change is smaller, multinest will have converged
        self.evidence_tolerance = evidence_tolerance
        self.mode_tolerance = mode_tolerance
        # importance nested sampling

        self.dir_polychord = polychord_path
        self._polychord_output = None
        self.resume = resume
        self.verbose = verbosity

    def compute_fit(self):
        self._polychord_output = None
        data = self._observed.spectrum
        datastd = self._observed.errorBar
        sqrtpi = np.sqrt(2*np.pi)

        ndim = len(self.fitting_parameters)

        def polychord_loglike(cube):
            # log-likelihood function called by polychord
            fit_params_container = np.array(
                [cube[i] for i in range(len(self.fitting_parameters))])
            chi_t = self.chisq_trans(fit_params_container, data, datastd)

            # print('---------START---------')
            # print('chi_t',chi_t)
            # print('LOG',loglike)
            loglike = -np.sum(np.log(datastd*sqrtpi)) - 0.5 * chi_t
            return loglike, [0.0]

        def polychord_uniform_prior(hypercube):
            # prior distributions called by polychord. Implements a uniform prior
            # converting parameters from normalised grid to uniform prior
            # print(type(cube))
            cube = [0.0]*ndim

            for idx, bounds in enumerate(self.fit_boundaries):
                # print(idx,self.fitting_parameters[idx])
                bound_min, bound_max = bounds
                cube[idx] = (hypercube[idx] *
                             (bound_max-bound_min)) + bound_min
                #print('CUBE idx',cube[idx])
            # print('-----------')
            return cube
        status = None

        datastd_mean = np.mean(datastd)

        settings = PolyChordSettings(ndim, 1)
        settings.nlive = ndim * 25
        settings.num_repeats = ndim * 5
        settings.do_clustering = self.do_clustering
        settings.num_repeats = ndim
        settings.precision_criterion = self.evidence_tolerance
        settings.logzero = -1e70
        settings.read_resume = self.resume
        settings.base_dir = self.dir_polychord
        settings.file_root = '1-'
        self.warning('Number of dimensions {}'.format(ndim))
        self.warning('Fitting parameters {}'.format(self.fitting_parameters))

        self.info('Beginning fit......')
        pypolychord.run_polychord(
            polychord_loglike, ndim, 1, settings, polychord_uniform_prior)
        self._polychord_output = self.store_polychord_solutions()
        print(self._polychord_output)

    def write_optimizer(self, output):
        opt = super().write_optimizer(output)

        # sampling efficiency (parameter, ...)
        opt.write_scalar('do_clustering', int(self.do_clustering))
        # run in constant efficiency mode
        # set log likelihood tolerance. If change is smaller, multinest will have converged
        opt.write_scalar('evidence_tolerance', self.evidence_tolerance)
        opt.write_scalar('mode_tolerance', self.mode_tolerance)
        # importance nested samplin

        return opt

    def write_fit(self, output):
        fit = super().write_fit(output)

        # if self._polychord_output:
        #     recursively_save_dict_contents_to_output(output,self._polychord_output)

        return fit

    def store_polychord_solutions(self):

        self.warning('Store the polychord results')
        NEST_out = {'solutions': {}}
        data = np.loadtxt(os.path.join(self.dir_polychord, '1-.txt'))

        # self.get_poly_cluster_number(self.dir_polychord)
        NEST_stats = self.get_poly_stats(self.dir_polychord)
        NEST_out['NEST_POLY_stats'] = NEST_stats
        NEST_out['global_logE'] = (NEST_out['NEST_POLY_stats']['global evidence'],
                                   NEST_out['NEST_POLY_stats']['global evidence error'])

        modes_array = []
        modes_weights = []
        num_fit_params = len(self.fit_names)
        num_clusters = 1

        if self.do_clustering:
            # if clustering is switched on, checking how many clusters exist
            num_clusters = self.get_poly_cluster_number(self.dir_polychord)
            print('NUM_CLUSTERS: ', num_clusters)
            if num_clusters == 1:
                # Get chains directly from file 1-.txt
                modes_array = [data[:, 2:num_fit_params+2]]
                modes_weights = [data[:, 0]]
            else:
                # cycle through clusters
                for midx in range(num_clusters):
                    data = np.loadtxt(os.path.join(
                        self.dir_polychord, 'clusters/1-_{0}.txt'.format(midx+1)))
                    modes_array.append(data[:, 2:num_fit_params+2])
                    modes_weights.append(data[:, 0])
        else:
            # Get chains directly from file 1-.txt
            modes_array = [data[:, 2:num_fit_params+2]]
            modes_weights = [data[:, 0]]

        modes_array = np.asarray(modes_array)
        modes_weights = np.asarray(modes_weights)

        for nmode in range(num_clusters):

            mydict = {'type': 'nest_poly',
                      'local_logE': (NEST_out['NEST_POLY_stats']['modes'][0]['local log-evidence'],  NEST_out['NEST_POLY_stats']['modes'][0]['local log-evidence error']),
                      'weights': np.asarray(modes_weights[nmode]),
                      'tracedata': modes_array[nmode],
                      'fit_params': {}}

            for idx, param_name in enumerate(self.fit_names):

                trace = modes_array[nmode][:, idx]
                q_16, q_50, q_84 = quantile_corner(trace, [0.16, 0.5, 0.84],
                                                   weights=np.asarray(modes_weights[nmode]))
                mydict['fit_params'][param_name] = {
                    'value': q_50,
                    'sigma_m': q_50-q_16,
                    'sigma_p': q_84-q_50,
                    'nest_map': NEST_stats['modes'][nmode]['maximum a posterior'][idx],
                    'nest_mean': NEST_stats['modes'][nmode]['mean'][idx],
                    'nest_sigma': NEST_stats['modes'][nmode]['sigma'][idx],
                    'trace': trace,
                }

            NEST_out['solutions']['solution{}'.format(nmode)] = mydict

        return NEST_out

    def get_poly_cluster_number(self, dir):
        import glob
        '''counts polychord cluster files in 'clusters' folder'''
        cluster_list = glob.glob(os.path.join(dir, 'clusters/1-*.txt'))
        c_idx = []
        for file in cluster_list:
            if file[-5].isdigit():
                c_idx.append(int(file[-5]))
        try:
            num = np.max(c_idx)
        except ValueError:
            num = 1
        return num

    def get_poly_stats(self, dir):
        '''replicates some of PyMultiNest.Analyzer for PolyChord'''
        stats = {}
        stats['modes'] = {}

        # re-count number of cluster files
        if self.do_clustering:
            num_clusters = self.get_poly_cluster_number(dir)
        else:
            num_clusters = 1

        nmode = 0  # mode index
        # open .stats file and reading global/local evidences
        with open(os.path.join(self.dir_polychord, '1-.stats')) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx == 8:  # skip to global evidence line
                    tmp_line = line.split()
                    gL_mu = tmp_line[2]
                    gL_sig = tmp_line[-1]
                    stats['global evidence'] = gL_mu
                    stats['global evidence error'] = gL_sig
                if idx > 13:  # skip to local evidence
                    tmp_line = line.split()
                    stats['modes'][nmode] = {}
                    stats['modes'][nmode]['local log-evidence'] = tmp_line[3]
                    stats['modes'][nmode]['local log-evidence error'] = tmp_line[5]
                    nmode += 1
                    # stopping reading file when clusters are exhausted
                    if idx == (13 + num_clusters):
                        break

        # opening cluster files (or global file if no clustering) and get MAP, mean, sigma for parameters
        if self.do_clustering:
            for midx in range(num_clusters):
                # cycling through cluster files
                data = np.loadtxt(os.path.join(
                    self.dir_polychord, 'clusters/1-_{0}.txt'.format(midx+1)))
                # find maximum likelihood index
                mL_idx = np.where(data[:, 1] == np.min(data[:, 1]))
                stats['modes'][midx]['maximum a posterior'] = {}
                stats['modes'][midx]['mean'] = {}
                stats['modes'][midx]['sigma'] = {}
                for idx in range(len(self.fit_names)):
                    # cycle through parameters
                    # maximum likelihood values
                    stats['modes'][midx]['maximum a posterior'][idx] = data[mL_idx, 2+idx]
                    # weighted average and sigma
                    mu, sig = weighted_avg_and_std(data[:, 2+idx], data[:, 0])
                    stats['modes'][midx]['mean'][idx] = mu
                    stats['modes'][midx]['sigma'][idx] = sig

        return stats

    def sample_parameters(self, solution):
        from taurex.util.util import random_int_iter
        solution_id = 'solution{}'.format(solution)
        samples = self._polychord_output['solutions'][solution_id]['tracedata']
        weights = self._polychord_output['solutions'][solution_id]['weights']

        for x in random_int_iter(samples.shape[0], self._sigma_fraction):
            w = weights[x]+1e-300

            yield samples[x, :], w

    def get_solution(self):
        names = self.fit_names
        opt_values = self.fit_values
        opt_map = self.fit_values
        solutions = [
            (k, v) for k, v in self._polychord_output['solutions'].items() if 'solution' in k]

        for k, v in solutions:
            solution_idx = int(k[8:])
            for p_name, p_value in v['fit_params'].items():
                if p_name in ('mu_derived',):
                    continue
                idx = names.index(p_name)
                opt_map[idx] = p_value['nest_map']
                opt_values[idx] = p_value['value']

            yield solution_idx, opt_map, opt_values, [
                ('fit_params', v['fit_params']),
                ('tracedata', v['tracedata']),
                ('weights', v['weights'])]
