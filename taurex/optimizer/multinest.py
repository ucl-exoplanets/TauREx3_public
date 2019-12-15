from .optimizer import Optimizer
import pymultinest
import numpy as np
import os
from taurex.mpi import get_rank, barrier
from taurex.util.util import read_table, read_error_line, read_error_into_dict, quantile_corner, recursively_save_dict_contents_to_output
import random
from taurex.util.util import weighted_avg_and_std
from taurex import OutputSize


class MultiNestOptimizer(Optimizer):

    def __init__(self, multi_nest_path=None, observed=None, model=None,
                 sampling_efficiency='parameter',
                 num_live_points=1500,
                 max_iterations=0,
                 search_multi_modes=True,
                 num_params_cluster=None,
                 maximum_modes=100,
                 constant_efficiency_mode=False,
                 evidence_tolerance=0.5,
                 mode_tolerance=-1e90,
                 importance_sampling=False,
                 resume=False,
                 multinest_prefix='1-',
                 verbose_output=True, sigma_fraction=0.1):
        super().__init__('Multinest', observed, model, sigma_fraction)

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
        self.multimodes = search_multi_modes
        # parameters on which to cluster, e.g. if nclust_par = 3, it will
        # cluster on the first 3 parameters only.
        # if ncluster_par = -1 it clusters on all parameters
        self.nclust_par = num_params_cluster
        # maximum number of modes
        self.max_modes = int(maximum_modes)
        # run in constant efficiency mode
        self.const_eff = constant_efficiency_mode
        # set log likelihood tolerance. If change is smaller, multinest will have converged
        self.evidence_tolerance = evidence_tolerance
        self.mode_tolerance = mode_tolerance
        # importance nested sampling
        self.imp_sampling = importance_sampling
        if self.imp_sampling:
            self.multimodes = False

        self.multinest_prefix = multinest_prefix

        self.dir_multinest = multi_nest_path
        if get_rank() == 0:

            if not os.path.exists(self.dir_multinest):
                self.info('Directory %s does not exist, creating',
                          self.dir_multinest)
                os.makedirs(self.dir_multinest)
        barrier()

        self.info('Found directory %s', self.dir_multinest)

        self.resume = resume
        self.verbose = verbose_output

    def compute_fit(self):

        data = self._observed.spectrum
        datastd = self._observed.errorBar
        sqrtpi = np.sqrt(2*np.pi)

        def multinest_loglike(cube, ndim, nparams):
            # log-likelihood function called by multinest
            fit_params_container = np.array(
                [cube[i] for i in range(len(self.fitting_parameters))])
            chi_t = self.chisq_trans(fit_params_container, data, datastd)

            # print('---------START---------')
            # print('chi_t',chi_t)
            # print('LOG',loglike)
            loglike = -np.sum(np.log(datastd*sqrtpi)) - 0.5 * chi_t
            # print(loglike)
            return loglike

        def multinest_uniform_prior(cube, ndim, nparams):
            # prior distributions called by multinest. Implements a uniform prior
            # converting parameters from normalised grid to uniform prior
            # print(type(cube))
            for idx, bounds in enumerate(self.fit_boundaries):
                # print(idx,self.fitting_parameters[idx])
                bound_min, bound_max = bounds
                cube[idx] = (cube[idx] * (bound_max-bound_min)) + bound_min
                #print('CUBE idx',cube[idx])
            # print('-----------')
        # status = None
        # def dump_call(nSamples,nlive,nPar,physLive,posterior,paramConstr,maxloglike,logZ,INSlogZ,logZerr,context):
        #    status = (nSamples,nlive,nPar,physLive,posterior,paramConstr,maxloglike,logZ,INSlogZ,logZerr,context)

        ndim = len(self.fitting_parameters)
        self.warning('Number of dimensions {}'.format(ndim))
        self.warning('Fitting parameters {}'.format(self.fitting_parameters))

        ncluster = self.nclust_par

        if isinstance(ncluster, float):
            ncluster = int(ncluster)

        if ncluster is not None and ncluster <= 0:
            ncluster = None

        if ncluster is None:
            self.nclust_par = ndim  # For writing to output later on

        self.info('Beginning fit......')
        pymultinest.run(LogLikelihood=multinest_loglike,
                        Prior=multinest_uniform_prior,
                        n_dims=ndim,
                        multimodal=self.multimodes,
                        n_clustering_params=ncluster,
                        max_modes=self.max_modes,
                        outputfiles_basename=os.path.join(self.dir_multinest,
                                                          self.multinest_prefix),
                        const_efficiency_mode=self.const_eff,
                        importance_nested_sampling=self.imp_sampling,
                        resume=self.resume,
                        verbose=self.verbose,
                        sampling_efficiency=self.sampling_eff,
                        evidence_tolerance=self.evidence_tolerance,
                        mode_tolerance=self.mode_tolerance,
                        n_live_points=self.n_live_points,
                        max_iter=self.max_iter
                        )

        self.info('Fit complete.....')

        self._multinest_output = self.store_nest_solutions()
        self.debug('Multinest output %s', self._multinest_output)

    def write_optimizer(self, output):
        opt = super().write_optimizer(output)

        # sampling efficiency (parameter, ...)
        opt.write_scalar('sampling_eff ', self.sampling_eff)
        # number of live points
        opt.write_scalar('num_live_points', self.n_live_points)
        # maximum no. of iterations (0=inf)
        opt.write_scalar('max_iterations', self.max_iter)
        # search for multiple modes
        opt.write_scalar('search_multimodes', self.multimodes)
        # parameters on which to cluster, e.g. if nclust_par = 3, it will cluster on the first 3 parameters only.
        # if ncluster_par = -1 it clusters on all parameters
        opt.write_scalar('nclust_parameter', self.nclust_par)
        # maximum number of modes
        opt.write_scalar('max_modes', self.max_modes)
        # run in constant efficiency mode
        opt.write_scalar('constant_efficiency', self.const_eff)
        # set log likelihood tolerance. If change is smaller, multinest will have converged
        opt.write_scalar('evidence_tolerance', self.evidence_tolerance)
        opt.write_scalar('mode_tolerance', self.mode_tolerance)
        # importance nested sampling
        opt.write_scalar('importance_sampling ', self.imp_sampling)

        return opt

    def write_fit(self, output):
        fit = super().write_fit(output)

        if self._multinest_output:
            recursively_save_dict_contents_to_output(
                output, self._multinest_output)

        return fit

    # Laziness brought us to this point
    # This function is so big and I cannot be arsed to rewrite this in a nicer way, if some angel does it
    # for me then I will buy them TWO beers.
    def store_nest_solutions(self):

        self.warning('Store the multinest results')
        NEST_out = {'solutions': {}}
        data = np.loadtxt(os.path.join(self.dir_multinest,
                                       '{}.txt'.format(self.multinest_prefix)))

        NEST_analyzer = pymultinest.Analyzer(n_params=len(
            self.fitting_parameters), outputfiles_basename=os.path.join(self.dir_multinest, self.multinest_prefix))
        NEST_stats = NEST_analyzer.get_stats()
        NEST_out['NEST_stats'] = NEST_stats
        NEST_out['global_logE'] = (
            NEST_out['NEST_stats']['global evidence'], NEST_out['NEST_stats']['global evidence error'])

        # Bypass if run in multimodes = False. Pymultinest.Analyser does not report means and sigmas in this case
        if len(NEST_out['NEST_stats']['modes']) == 0:
            with open(os.path.join(self.dir_multinest, '{}stats.dat'.format(self.multinest_prefix))) as file:
                lines = file.readlines()
            stats = {'modes': []}
            read_error_into_dict(lines[0], stats)

            text = ''.join(lines[2:])
            # without INS:
            parts = text.split("\n\n")
            mode = {'index': 0}

            modelines = parts[0]
            t = read_table(modelines, title='Parameter')
            mode['mean'] = t[:, 1].tolist()
            mode['sigma'] = t[:, 2].tolist()

            mode['maximum'] = read_table(parts[1], title='Parameter', d=None)[
                :, 1].tolist()
            mode['maximum a posterior'] = read_table(
                parts[2], title='Parameter', d=None)[:, 1].tolist()

            if 'Nested Sampling Global Log-Evidence'.lower() in stats:
                mode['Local Log-Evidence'.lower()
                     ] = stats['Nested Sampling Global Log-Evidence'.lower()]
                mode['Local Log-Evidence error'.lower()
                     ] = stats['Nested Sampling Global Log-Evidence error'.lower()]
            else:
                mode['Local Log-Evidence'.lower()
                     ] = stats['Nested Importance Sampling Global Log-Evidence'.lower()]
                mode['Local Log-Evidence error'.lower()
                     ] = stats['Nested Importance Sampling Global Log-Evidence error'.lower()]

            mode['Strictly Local Log-Evidence'.lower()
                 ] = mode['Local Log-Evidence'.lower()]
            mode['Strictly Local Log-Evidence error'.lower()
                 ] = mode['Local Log-Evidence error'.lower()]

            NEST_out['NEST_stats']['modes'] = [mode]

        modes = []
        modes_weights = []
        chains = []
        chains_weights = []

        if self.multimodes:

            # separate modes. get individual samples for each mode

            # get parameter values and sample probability (=weight) for each mode
            with open(os.path.join(self.dir_multinest, '{}post_separate.dat'.format(self.multinest_prefix))) as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if idx > 2:  # skip the first two lines
                        if lines[idx-1] == '\n' and lines[idx-2] == '\n':
                            modes.append(chains)
                            modes_weights.append(chains_weights)
                            chains = []
                            chains_weights = []
                    chain = [float(x) for x in line.split()[2:]]
                    if len(chain) > 0:
                        chains.append(chain)
                        chains_weights.append(float(line.split()[0]))
                modes.append(chains)
                modes_weights.append(chains_weights)
            modes_array = []
            for mode in modes:
                mode_array = np.zeros((len(mode), len(mode[0])))
                for idx, line in enumerate(mode):
                    mode_array[idx, :] = line
                modes_array.append(mode_array)
        else:
            # not running in multimode. Get chains directly from file 1-.txt
            modes_array = [data[:, 2:]]
            chains_weights = [data[:, 0]]
            modes_weights.append(chains_weights[0])
            modes = [0]

        modes_weights = np.asarray(modes_weights)
        for nmode in range(len(modes)):
            self.debug('Nmode: {}'.format(nmode))

            mydict = {'type': 'nest',
                      'local_logE': (NEST_out['NEST_stats']['modes'][0]['local log-evidence'],  NEST_out['NEST_stats']['modes'][0]['local log-evidence error']),
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
                    'mean': NEST_stats['modes'][nmode]['mean'][idx],
                    'nest_sigma': NEST_stats['modes'][nmode]['sigma'][idx],
                    'trace': trace,
                }

            NEST_out['solutions']['solution{}'.format(nmode)] = mydict

        return NEST_out

    def generate_solution(self, output_size=OutputSize.heavy):

        solution = super().generate_solution(output_size=output_size)

        solution['GlobalStats'] = self._multinest_output['NEST_stats']
        return solution

    def sample_parameters(self, solution):
        from taurex.util.util import random_int_iter
        solution_id = 'solution{}'.format(solution)
        samples = self._multinest_output['solutions'][solution_id]['tracedata']
        weights = self._multinest_output['solutions'][solution_id]['weights']

        for x in random_int_iter(samples.shape[0], self._sigma_fraction):
            w = weights[x]+1e-300

            yield samples[x, :], w

    def get_solution(self):
        names = self.fit_names
        opt_values = self.fit_values
        opt_map = self.fit_values
        solutions = [
            (k, v) for k, v in self._multinest_output['solutions'].items() if 'solution' in k]
        for k, v in solutions:
            solution_idx = int(k[8:])
            for p_name, p_value in v['fit_params'].items():
                if p_name in ('mu_derived',):
                    continue
                idx = names.index(p_name)
                opt_map[idx] = p_value['nest_map']
                opt_values[idx] = p_value['value']

            yield solution_idx, opt_map, opt_values, [
                ('Statistics', {'local log-evidence': self._multinest_output['NEST_stats']['modes'][solution_idx]['local log-evidence'],
                                'local log-evidence error': self._multinest_output['NEST_stats']['modes'][solution_idx]['local log-evidence error']}),
                ('fit_params', v['fit_params']),
                ('tracedata', v['tracedata']),
                ('weights', v['weights'])]
