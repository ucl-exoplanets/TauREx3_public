from taurex.log import Logger, disableLogging, enableLogging
import numpy as np
import math
from taurex import OutputSize


class Optimizer(Logger):
    """
    A base class that handles fitting and optimization of forward models.
    The class handles the compiling and management of fitting parameters in 
    forward models, in its current form it cannot fit and requires a class 
    derived from it to implement the :func:`compute_fit` function.

    Parameters
    ----------
    name:  str
        Name to be used in logging

    observed : :class:`~taurex.data.spectrum.spectrum.BaseSpectrum`, optional
        See :func:`set_observed`

    model : :class:`~taurex.model.model.ForwardModel`, optional
        See :func:`set_model`

    sigma_fraction: float, optional
        Fraction of weights to use in computing the error. (Default: 0.1)

    """

    def __init__(self, name, observed=None, model=None, sigma_fraction=0.1):
        super().__init__(name)

        self.set_model(model)
        self.set_observed(observed)
        self._model_callback = None
        self._sigma_fraction = sigma_fraction

    def set_model(self, model):
        """
        Sets the model to be optimized/fit

        Parameters
        ----------
        model : :class:`~taurex.model.model.ForwardModel`
            The forward model we wish to optimize

        """
        self._model = model

    def set_observed(self, observed):
        """
        Sets the observation to optimize the model to

        Parameters
        ----------
        observed : :class:`~taurex.data.spectrum.spectrum.BaseSpectrum`
            Observed spectrum we will optimize to

        """
        self._observed = observed
        if observed is not None:
            self._binner = observed.create_binner()

    def compile_params(self):
        """ 


        Goes through and compiles all parameters within the model that
        we will be retrieving. Called before :func:`compute_fit`


        """
        self.info('Initializing parameters')
        self.fitting_parameters = []
        # param_name,param_latex,
        #                 fget.__get__(self),fset.__get__(self),
        #                         default_fit,default_bounds
        for params in self._model.fittingParameters.values():
            name, latex, fget, fset, mode, to_fit, bounds = params

            self.debug('Checking fitting parameter {}'.format(params))
            if to_fit:
                self.fitting_parameters.append(params)

        self.info('-------FITTING---------------')
        self.info('Parameters to be fit:')
        for params in self.fitting_parameters:
            name, latex, fget, fset, mode, to_fit, bounds = params
            self.info('{}: Value: {} Mode:{} Boundaries:{}'.format(
                name, fget(), mode, bounds))

    def update_model(self, fit_params):
        """
        Updates the model with new parameters

        Parameters
        ----------
        fit_params : :obj:`list`
            A list of new values to apply to the model. The list of values are
            assumed to be in the same order as the parameters given by 
            :func:`fit_names`


        """

        for value, param in zip(fit_params, self.fitting_parameters):
            name, latex, fget, fset, mode, to_fit, bounds = param
            if mode == 'log':
                value = 10**value
            fset(value)

    @property
    def fit_values_nomode(self):
        """ 

        Returns a list of the current values of a fitting parameter. 
        Regardless of the ``mode`` setting

        Returns
        -------
        :obj:`list`:
            List of each value of a fitting parameter

        """

        return [c[2]() for c in self.fitting_parameters]

    @property
    def fit_values(self):
        """ 

        Returns a list of the current values of a fitting parameter. This 
        respects the ``mode`` setting

        Returns
        -------
        :obj:`list`:
            List of each value of a fitting parameter

        """

        return [c[2]() if c[4] == 'linear' else math.log10(c[2]()) 
                for c in self.fitting_parameters]

    @property
    def fit_boundaries(self):
        """

        Returns the fitting boundaries of the parameter

        Returns
        -------
        :obj:`list`:
            List of boundaries for each fitting parameter. It takes the form of
            a python :obj:`tuple` with the form 
            ( ``bound_min`` , ``bound_max`` )

        """
        return [c[-1] if c[4] == 'linear'
                else (math.log10(c[-1][0]), math.log10(c[-1][1]))
                for c in self.fitting_parameters]

    @property
    def fit_names(self):
        """

        Returns the names of the model parameters we will be fitting

        Returns
        -------
        :obj:`list`:
            List of names of parameters that will be fit

        """
        return [c[0] if c[4] == 'linear' else 'log_{}'.format(c[0])
                for c in self.fitting_parameters]

    @property
    def fit_latex(self):
        """

        Returns the names of the parameters in LaTeX format

        Returns
        -------
        :obj:`list`
            List of parameter names in LaTeX format


        """

        return [c[1] if c[4] == 'linear' else 'log({})'.format(c[1])
                for c in self.fitting_parameters]

    def enable_fit(self, parameter):
        """
        Enables fitting of the parameter

        Parameters
        ----------
        parameter : str
            Name of the parameter we want to fit

        """

        name, latex, fget, fset, mode, to_fit, bounds = \
            self._model.fittingParameters[parameter]

        to_fit = True

        self._model.fittingParameters[parameter] = (
            name, latex, fget, fset, mode, to_fit, bounds)

    def disable_fit(self, parameter):
        """
        Disables fitting of the parameter

        Parameters
        ----------
        parameter : str
            Name of the parameter we do not want to fit

        """
        name, latex, fget, fset, mode, to_fit, bounds = \
            self._model.fittingParameters[parameter]

        to_fit = False

        self._model.fittingParameters[parameter] = (
            name, latex, fget, fset, mode, to_fit, bounds)

    def set_boundary(self, parameter, new_boundaries):
        """
        Sets the boundary of the parameter

        Parameters
        ----------
        parameter : str
            Name of the parameter we want to change

        new_boundaries : tuple of float
            New fitting boundaries, with the form
            ( ``bound_min`` , ``bound_max`` ). These should
            not take into account the ``mode`` setting of a fitting parameter.


        """
        name, latex, fget, fset, mode, to_fit, bounds = \
            self._model.fittingParameters[parameter]

        bounds = new_boundaries

        self._model.fittingParameters[parameter] = (
            name, latex, fget, fset, mode, to_fit, bounds)

    def set_factor_boundary(self, parameter, factors):
        """
        Sets the boundary of the parameter based on a factor

        Parameters
        ----------
        parameter : str
            Name of the parameter we want to change

        factor : tuple of float
            To be written



        """

        name, latex, fget, fset, mode, to_fit, bounds = \
            self._model.fittingParameters[parameter]

        value = fget()

        new_boundaries = factors[0]*value, factors[1]*value

        bounds = new_boundaries
        self._model.fittingParameters[parameter] = (
            name, latex, fget, fset, mode, to_fit, bounds)

    def set_mode(self, parameter, new_mode):
        """
        Sets the fitting mode of a parameter

        Parameters
        ----------
        parameter : str
            Name of the parameter we want to change

        new_mode : ``linear`` or ``log``
            Sets whether the parameter is fit in linear or log space



        """
        new_mode = new_mode.lower()

        name, latex, fget, fset, mode, to_fit, bounds = \
            self._model.fittingParameters[parameter]

        if new_mode not in ('log', 'linear',):
            self.error('Incorrect mode set for fit parameter,')
            raise ValueError

        self._model.fittingParameters[parameter] = (
            name, latex, fget, fset, new_mode, to_fit, bounds)

    def chisq_trans(self, fit_params, data, datastd):
        """

        Computes the Chi-Squared between the forward model and
        observation. The steps taken are:
            1. Forward model (FM) is updated with :func:`update_model`
            2. FM is then computed at its native grid then binned.
            3. Chi-squared between FM and observation is computed


        Parameters
        ----------
        fit_params : list of parameter values
            List of new parameter values to update the model

        data : obj:`ndarray`
            Observed spectrum

        datastd : obj:`ndarray`
            Observed spectrum error


        Returns
        -------
        float :
            chi-squared


        """
        from taurex.exceptions import InvalidModelException
        self.update_model(fit_params)

        obs_bins = self._observed.wavenumberGrid

        try:
            _, final_model, _, _ = self._binner.bin_model(
                self._model.model(wngrid=obs_bins))
        except InvalidModelException:
            return 1e100

        res = (data.ravel() - final_model.ravel()) / datastd.ravel()
        res = np.nansum(res*res)
        if res == 0:
            res = np.nan

        return res

    def compute_fit(self):
        """
        Unimplemented. When inheriting this should be overwritten
        to perform the actual fit.

        Raises
        ------
        NotImplementedError
            Raised when a derived class does override this function

        """
        raise NotImplementedError

    def fit(self, output_size=OutputSize.heavy):
        import time
        """

        Performs fit.

        """
        from tabulate import tabulate
        self.compile_params()

        fit_names = self.fit_names
        fit_boundaries = self.fit_boundaries

        fit_min = [x[0] for x in fit_boundaries]
        fit_max = [x[1] for x in fit_boundaries]

        fit_values = self.fit_values
        self.info('')
        self.info('-------------------------------------')
        self.info('------Retrieval Parameters-----------')
        self.info('-------------------------------------')
        self.info('')
        self.info('Dimensionality of fit: %s', len(fit_names))
        self.info('')

        output = tabulate(zip(fit_names, fit_values, fit_min, fit_max),
                          headers=['Param', 'Value', 'Bound-min', 'Bound-max'])

        self.info('\n%s\n\n', output)
        self.info('')
        start_time = time.time()
        disableLogging()
        # Compute fit here
        self.compute_fit()

        enableLogging()
        end_time = time.time()
        self.info('Sampling time %s s', end_time-start_time)
        solution = self.generate_solution(output_size=output_size)
        self.info('')
        self.info('-------------------------------------')
        self.info('------Final results------------------')
        self.info('-------------------------------------')
        self.info('')
        self.info('Dimensionality of fit: %s', len(fit_names))
        self.info('')
        for idx, optimized_map, optimized_median, values in self.get_solution():
            self.info('\n%s', '---Solution {}------'.format(idx))
            output = tabulate(zip(fit_names, optimized_map, optimized_median), headers=[
                              'Param', 'MAP', 'Median'])
            self.info('\n%s\n\n', output)
        return solution

    def write_optimizer(self, output):
        """

        Writes optimizer settings under the 'Optimizer' heading in an
        output file

        Parameters
        ----------
        output:  :class:`~taurex.output.output.Output` or :class:`~taurex.output.output.OutputGroup`
            Group (or root) in output file to write to

        Returns
        -------
        :class:`~taurex.output.output.Output` or :class:`~taurex.output.output.OutputGroup`
            Group (or root) in output file written to

        """
        output.write_string('optimizer', self.__class__.__name__)
        output.write_string_array('fit_parameter_names', self.fit_names)
        output.write_string_array('fit_parameter_latex', self.fit_latex)
        output.write_array('fit_boundary_low', np.array(
            [x[0] for x in self.fit_boundaries]))
        output.write_array('fit_boundary_high', np.array(
            [x[1] for x in self.fit_boundaries]))
        return output

    def write_fit(self, output):
        """
        Writes basic fitting parameters into output

        Parameters
        ----------
        output : :class:`~taurex.output.output.Output` or :class:`~taurex.output.output.OutputGroup`
            Group (or root) in output file to write to

        Returns
        -------
        :class:`~taurex.output.output.Output` or :class:`~taurex.output.output.OutputGroup`
            Group (or root) in output file written to

        """
        fit = output.create_group('FitParams')
        fit.write_string('fit_format', self.__class__.__name__)
        fit.write_string_array('fit_parameter_names', self.fit_names)
        fit.write_string_array('fit_parameter_latex', self.fit_latex)
        fit.write_array('fit_boundary_low', np.array(
            [x[0] for x in self.fit_boundaries]))
        fit.write_array('fit_boundary_high', np.array(
            [x[1] for x in self.fit_boundaries]))

        # This is the last sampled value ... should not be recorded to avoid confusion.
        # fit.write_list('fit_parameter_values',self.fit_values)
        # fit.write_list('fit_parameter_values_nomode',self.fit_values_nomode)
        return output

    def generate_profiles(self, solution, binning):
        from taurex import mpi
        """Generates sigma plots for profiles"""

        sample_list = None

        if mpi.get_rank() == 0:
            sample_list = list(self.sample_parameters(solution))

        sample_list = mpi.broadcast(sample_list)

        self.debug('We all got %s', sample_list)

        self.info('------------Variance generation step------------------')

        self.info('We are sampling %s points for the profiles',
                  len(sample_list))

        rank = mpi.get_rank()
        size = mpi.nprocs()

        enableLogging()

        self.info('I will only iterate through partitioned %s '
                  'points (the rest is in parallel)', len(sample_list)//size)

        disableLogging()
        count = 0

        def sample_iter():
            for parameters, weight in sample_list[rank::size]:
                self.update_model(parameters)
                enableLogging()
                if rank == 0 and count % 10 == 0 and count > 0:

                    self.info('Progress {}%'.format(
                        count*100.0 / (len(sample_list)/size)))
                disableLogging()
                yield weight

        return self._model.compute_error(sample_iter, wngrid=binning,
                                         binner=self._binner)

    def generate_solution(self, output_size=OutputSize.heavy):
        """
        Generates a dictionary with all solutions and other useful parameters
        """
        from taurex.util.output import generate_profile_dict, \
            store_contributions

        solution_dict = {}

        self.info('Generating spectra and profiles')

        # Loop through each solution, grab optimized parameters and anything 
        # else we want to store
        for solution, optimized_map, \
                optimized_median, values in self.get_solution():

            self.info('Computing solution %s', solution)
            sol_values = {}

            # Include extra stuff we might want to store (provided by the child)
            for k, v in values:
                sol_values[k] = v

            # Update the model with optimized map values
            self.update_model(optimized_map)

            opt_result = self._model.model(cutoff_grid=False)  # Run the model

            sol_values['Spectra'] = self._binner.generate_spectrum_output(
                opt_result, output_size=output_size)

            sol_values['Spectra']['Contributions'] = store_contributions(
                self._binner, self._model, output_size=output_size-3)

            # Update with the optimized median
            self.update_model(optimized_median)

            self._model.model(cutoff_grid=False)

            # Store profiles here
            sol_values['Profiles'] = generate_profile_dict(self._model)
            profile_dict, spectrum_dict = self.generate_profiles(
                solution, self._observed.wavenumberGrid)

            for k, v in profile_dict.items():
                sol_values['Profiles'][k] = v

            for k, v in spectrum_dict.items():
                sol_values['Spectra'][k] = v

            solution_dict['solution{}'.format(solution)] = sol_values

        # Compute mu derived
        for solution, optimized_map, \
                optimized_median, values in self.get_solution():

            mu = self.compute_mu_derived_trace(solution)

            solution_dict['solution{}'.format(
                solution)]['fit_params']['mu_derived'] = mu

        return solution_dict

    def compute_mu_derived_trace(self, solution):
        from taurex.util.util import quantile_corner
        from taurex.constants import AMU
        sigma_frac = self._sigma_fraction
        self._sigma_fraction = 1.0
        mu_trace = []
        weights = []
        self.info('Computing derived mu......')
        disableLogging()
        for parameters, weight in self.sample_parameters(solution):
            self.update_model(parameters)
            self._model.initialize_profiles()
            mu_trace.append(self._model.chemistry.muProfile[0]/AMU)
            weights.append(weight)
        enableLogging()

        self.info('Done!')

        self._sigma_fraction = sigma_frac

        q_16, q_50, q_84 = \
            quantile_corner(np.array(mu_trace), [0.16, 0.5, 0.84],
                            weights=np.array(weights))

        mean = np.average(mu_trace, weights=weights, axis=0)

        mu_derived = {
            'value': q_50,
            'sigma_m': q_50-q_16,
            'sigma_p': q_84-q_50,
            'trace': mu_trace,
            'mean': mean
        }
        return mu_derived

    def sample_parameters(self, solution):
        """
        **Requires implementation***

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
        raise NotImplementedError

    def get_solution(self):
        """
        ** Requires implementation **

        Generator for solutions and their
        median and MAP values

        Yields
        ------

        solution_no: int
            Solution number
        
        map: :obj:`array`
            Map values
        
        median: :obj:`array`
            Median values
        
        extra: :obj:`list`
            List of tuples of extra information to store.
            Must be of form ``(name, data)``
            
        

        """
        raise NotImplementedError

    def write(self, output):
        """
        Creates 'Optimizer'
        them respectively



        Parameters
        ----------
        output : :class:`~taurex.output.output.Output` or :class:`~taurex.output.output.OutputGroup`
            Group (or root) in output file to write to



        """
        opt = output.create_group('Optimizer')
        self.write_optimizer(opt)
