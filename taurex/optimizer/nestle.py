from .optimizer import Optimizer
import nestle
import numpy as np
import time
from taurex.util.util import quantile_corner, \
                                recursively_save_dict_contents_to_output


class NestleOptimizer(Optimizer):
    """ An optimizer that uses the `nestle <http://kylebarbary.com/nestle/>`_ library
    to perform optimization. 

    Parameters
    ----------
    observed : :class:`~taurex.data.spectrum.spectrum.BaseSpectrum`, optional
        Observed spectrum to optimize to

    model : :class:`~taurex.model.model.ForwardModel`, optional
        Forward model to optimize

    num_live_points : int, optional
        Number of live points to use in sampling

    method : ``classic``, ``single`` or ``multi``
        Nested sampling method to use. ``classic`` uses MCMC exploration,
        ``single`` uses a single ellipsoid and ``multi`` uses multiple ellipsoids (similar to Multinest) 

    tol : float
        Evidence tolerance value to stop the fit. This is based on an estimate of the remaining prior volumes
        contribution to the evidence.

    sigma_fraction: float, optional
        Fraction of weights to use in computing the error. (Default: 0.1)

    """

    def __init__(self, observed=None, model=None, num_live_points=1500, method='multi', tol=0.5, sigma_fraction=0.1):
        super().__init__('Nestle', observed, model, sigma_fraction)
        self._nlive = int(num_live_points)    # number of live points
        self._method = method  # use MutliNest algorithm

        self._tol = tol       # the stopping criterion
        self._nestle_output = None

    @property
    def tolerance(self):
        return self._tol

    @tolerance.setter
    def tolerance(self, value):
        self._tol = value

    @property
    def numLivePoints(self):
        return self._nlive

    @numLivePoints.setter
    def numLivePoints(self, value):
        self._nlive = value

    def compute_fit(self):
        """

        Computes the fit using nestle

        """
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

            for idx, bounds in enumerate(self.fit_boundaries):
                bound_min, bound_max = bounds
                cube.append((theta[idx] * (bound_max-bound_min)) + bound_min)

            return tuple(cube)

        ndim = len(self.fitting_parameters)
        self.warning('Beginning fit......')
        ndims = ndim        # two parameters

        t0 = time.time()

        res = nestle.sample(nestle_loglike, nestle_uniform_prior, ndims, method='multi',
                            npoints=self.numLivePoints, dlogz=self.tolerance, callback=nestle.print_progress)
        t1 = time.time()

        timenestle = (t1-t0)

        print(res.summary())

        self.warning("Time taken to run 'Nestle' is %s seconds", timenestle)

        self.warning('Fit complete.....')

        self._nestle_output = self.store_nestle_output(res)

    def sample_parameters(self, solution):
        """
        Read traces and weights and return
        a random ``sigma_fraction`` sample of them

        Parameters
        ----------
        solution:
            a solution output from sampler

        Yields
        ------
        traces: :obj:`array`
            Traces of a particular sample

        weight: float
            Weight of sample
        
        """
        from taurex.util.util import random_int_iter
        samples = self._nestle_output['solution']['samples']
        weights = self._nestle_output['solution']['weights']

        for x in random_int_iter(samples.shape[0], self._sigma_fraction):
            w = weights[x]+1e-300

            yield samples[x, :], w

    def get_solution(self):
        """

        Generator for solutions and their
        median and MAP values

        Yields
        ------

        solution_no: int
            Solution number (always 0)

        map: :obj:`array`
            Map values

        median: :obj:`array`
            Median values

        extra: :obj:`list`
            Returns Statistics, fitting_params, raw_traces and
            raw_weights

        """

        names = self.fit_names
        opt_map = self.fit_values
        opt_values = self.fit_values
        for k, v in self._nestle_output['solution']['fitparams'].items():
            if k in ('mu_derived',):
                continue
            idx = names.index(k)
            opt_map[idx] = v['map']
            opt_values[idx] = v['value']

        yield 0, opt_map, opt_values, [('Statistics', self._nestle_output['Stats']),
                                       ('fit_params',
                                        self._nestle_output['solution']['fitparams']),
                                       ('tracedata',
                                        self._nestle_output['solution']['samples']),
                                       ('weights', self._nestle_output['solution']['weights'])]

    def write_optimizer(self, output):
        opt = super().write_optimizer(output)

        # number of live points
        opt.write_scalar('num_live_points', self._nlive)
        # maximum no. of iterations (0=inf)
        opt.write_string('method', self._method)
        # search for multiple modes
        opt.write_scalar('tol', self._tol)

        return opt

    def write_fit(self, output):
        fit = super().write_fit(output)

        if self._nestle_output:
            recursively_save_dict_contents_to_output(
                output, self._nestle_output)

        return fit

    def store_nestle_output(self, result):
        """
        This turns the output fron nestle into a dictionary that can
        be output by Taurex

        Parameters
        ----------
        result: :obj:`dict`
            Result from a nestle sample call

        Returns
        -------
        dict
            Formatted dictionary for output

        """

        from tabulate import tabulate

        nestle_output = {}
        nestle_output['Stats'] = {}
        nestle_output['Stats']['Log-Evidence'] = result.logz
        nestle_output['Stats']['Log-Evidence-Error'] = result.logzerr
        nestle_output['Stats']['Peakiness'] = result.h

        fit_param = self.fit_names

        samples = result.samples
        weights = result.weights

        mean, cov = nestle.mean_and_cov(samples, weights)
        nestle_output['solution'] = {}
        nestle_output['solution']['samples'] = samples
        nestle_output['solution']['weights'] = weights
        nestle_output['solution']['covariance'] = cov
        nestle_output['solution']['fitparams'] = {}

        max_weight = weights.argmax()

        table_data = []

        for idx, param_name in enumerate(fit_param):
            param = {}
            param['mean'] = mean[idx]
            param['sigma'] = cov[idx]
            trace = samples[:, idx]
            q_16, q_50, q_84 = quantile_corner(trace, [0.16, 0.5, 0.84],
                                               weights=np.asarray(weights))
            param['value'] = q_50
            param['sigma_m'] = q_50-q_16
            param['sigma_p'] = q_84-q_50
            param['trace'] = trace
            param['map'] = trace[max_weight]
            table_data.append((param_name, q_50, q_50-q_16))

            nestle_output['solution']['fitparams'][param_name] = param

        return nestle_output
