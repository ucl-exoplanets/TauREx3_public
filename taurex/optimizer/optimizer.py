from taurex.log import Logger,disableLogging,enableLogging
import numpy as np
from taurex.output.writeable import Writeable
import math
from taurex import OutputSize
from taurex.mpi import allgather
class Optimizer(Logger):
    """
    A base class that handles fitting and optimization of forward models.
    The class handles the compiling and management of fitting parameters in 
    forward models, in its current form it cannot fit and requires a class derived from it
    to implement the :func:`compute_fit` function.

    Parameters
    ----------
    name:  str
        Name to be used in logging

    observed : :class:`~taurex.data.spectrum.spectrum.BaseSpectrum`, optional
        See :func:`set_observed`
    
    model : :class:`~taurex.model.model.ForwardModel`, optional
        See :func:`set_model`


    """

    def __init__(self,name,observed=None,model=None,sigma_fraction=0.1):
        super().__init__(name)

        self.set_model(model)
        self.set_observed(observed)
        self._model_callback = None
        self._sigma_fraction = sigma_fraction

    def set_model(self,model):
        """
        Sets the model to be optimized/fit

        Parameters
        ----------
        model : :class:`~taurex.model.model.ForwardModel`
            The forward model we wish to optimize
        
        """
        self._model = model

    def set_observed(self,observed):
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
        self.fitting_parameters=[]
        # param_name,param_latex,
        #                 fget.__get__(self),fset.__get__(self),
        #                         default_fit,default_bounds
        for params in self._model.fittingParameters.values():
            name,latex,fget,fset,mode,to_fit,bounds = params

            self.debug('Checking fitting parameter {}'.format(params))
            if to_fit:
                # #To make sure of any conversions going on lets get the current value
                # c_v = fget()

                # #Get the boundaries
                # bound_min= bounds[0]
                # bound_max = bounds[1]

                # #set them to the internal model
                # fset(bound_min)
                # bound_min = fget()

                # fset(bound_max)
                # bound_max = fget()

                # fset(c_v)

                self.fitting_parameters.append(params)
        
        self.info('-------FITTING---------------')
        self.info('Parameters to be fit:')
        for params in self.fitting_parameters:
            name,latex,fget,fset,mode,to_fit,bounds = params
            self.info('{}: Value: {} Mode:{} Boundaries:{}'.format(name,fget(),mode,bounds))


    def update_model(self,fit_params):
        """
        Updates the model with new parameters
        
        Parameters
        ----------
        fit_params : :obj:`list`
            A list of new values to apply to the model. The list of values are
            assumed to be in the same order as the parameters given by :func:`fit_names`

        
        """

        for value,param in zip(fit_params,self.fitting_parameters):
            name,latex,fget,fset,mode,to_fit,bounds = param
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

        return [c[2]() if c[4]=='linear' else math.log10(c[2]())for c in self.fitting_parameters]

    @property
    def fit_boundaries(self):
        """ 
        
        Returns the fitting boundaries of the parameter

        Returns
        -------
        :obj:`list`:
            List of boundaries for each fitting parameter. It takes the form of
            a python :obj:`tuple` with the form ( ``bound_min`` , ``bound_max`` )
        
        """
        return [c[-1] if c[4]=='linear' else (math.log10(c[-1][0]),math.log10(c[-1][1])) for c in self.fitting_parameters]


    @property
    def fit_names(self):
        """ 
        
        Returns the names of the model parameters we will be fitting

        Returns
        -------
        :obj:`list`:
            List of names of parameters that will be fit
        
        """
        return [c[0] if c[4]=='linear' else 'log_{}'.format(c[0]) for c in self.fitting_parameters]

    @property
    def fit_latex(self):
        """ 
        
        Returns the names of the parameters in LaTeX format

        Returns
        -------
        :obj:`list`
            List of parameter names in LaTeX format
        

        """

        return [c[1] if c[4]=='linear' else 'log({})'.format(c[1]) for c in self.fitting_parameters]

    def enable_fit(self,parameter):
        """
        Enables fitting of the parameter

        Parameters
        ----------
        parameter : str
            Name of the parameter we want to fit
        
        """


        name,latex,fget,fset,mode,to_fit,bounds = self._model.fittingParameters[parameter]
        to_fit = True
        self._model.fittingParameters[parameter]= (name,latex,fget,fset,mode,to_fit,bounds)

    def disable_fit(self,parameter):
        """
        Disables fitting of the parameter

        Parameters
        ----------
        parameter : str
            Name of the parameter we do not want to fit
        
        """
        name,latex,fget,fset,mode,to_fit,bounds = self._model.fittingParameters[parameter]
        to_fit = False
        self._model.fittingParameters[parameter]= (name,latex,fget,fset,mode,to_fit,bounds)
    
    def set_boundary(self,parameter,new_boundaries):
        """
        Sets the boundary of the parameter

        Parameters
        ----------
        parameter : str
            Name of the parameter we want to change
        
        new_boundaries : tuple of float
            New fitting boundaries, with the form ( ``bound_min`` , ``bound_max`` ). These
            should not take into account the ``mode`` setting of a fitting parameter.

        
        """
        name,latex,fget,fset,mode,to_fit,bounds = self._model.fittingParameters[parameter]
        bounds = new_boundaries
        self._model.fittingParameters[parameter]= (name,latex,fget,fset,mode,to_fit,bounds)


    def set_factor_boundary(self,parameter,factors):
        """
        Sets the boundary of the parameter based on a factor

        Parameters
        ----------
        parameter : str
            Name of the parameter we want to change
        
        factor : tuple of float
            To be written


          
        """

        name,latex,fget,fset,mode,to_fit,bounds = self._model.fittingParameters[parameter]

        value = fget()

        # if mode == 'log':
        #     log_value = math.log10(value)

        #     new_boundaries = 10**(log_value/factors[0]),10**(log_value*factors[1])
        # else:
        #     
        new_boundaries = factors[0]*value,factors[1]*value
        
        bounds = new_boundaries
        self._model.fittingParameters[parameter]= (name,latex,fget,fset,mode,to_fit,bounds)

    def set_mode(self,parameter,new_mode):
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
        name,latex,fget,fset,mode,to_fit,bounds = self._model.fittingParameters[parameter]
        if not new_mode in ('log','linear',):
            self.error('Incorrect mode set for fit parameter,')
            raise ValueError

        self._model.fittingParameters[parameter]= (name,latex,fget,fset,new_mode,to_fit,bounds)       


    def chisq_trans(self, fit_params,data,datastd):
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
        from taurex.util import bindown
        import numexpr as ne
        from taurex.exceptions import InvalidModelException
        self.update_model(fit_params)

        obs_bins= self._observed.wavenumberGrid

        

        try:
            _,final_model,_,_ = self._binner.bin_model(self._model.model(obs_bins))
        except InvalidModelException:
            return 1e100

            
        res = (data.flatten() - final_model.flatten()) / datastd.flatten()
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


    def fit(self,output_size=OutputSize.heavy):
        from taurex.log import setLogLevel
        import logging
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
        self.info('Dimensionality of fit: %s',len(fit_names))
        self.info('')
        output = tabulate(zip(fit_names,fit_values,fit_min,fit_max), headers=['Param', 'Value','Bound-min', 'Bound-max'])
        self.info('\n%s\n\n',output)
        self.info('')

        disableLogging()
        self.compute_fit()
        enableLogging()
        return  self.generate_solution(output_size=output_size)



    def write_optimizer(self,output):
        """

        Writes optimizer settings under the 'Optimizer' heading in an output file

        Parameters
        ----------
        output:  :class:`~taurex.output.output.Output` or :class:`~taurex.output.output.OutputGroup`
            Group (or root) in output file to write to

        Returns
        -------
        :class:`~taurex.output.output.Output` or :class:`~taurex.output.output.OutputGroup`
            Group (or root) in output file written to

        """
        output.write_string('optimizer',self.__class__.__name__)
        output.write_string_array('fit_parameter_names',self.fit_names)
        output.write_string_array('fit_parameter_latex',self.fit_latex)
        output.write_array('fit_boundary_low',np.array([x[0] for x in self.fit_boundaries]))
        output.write_array('fit_boundary_high',np.array([x[1] for x in self.fit_boundaries]))
        return output
    
    def write_fit(self,output):
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
        fit.write_string('fit_format',self.__class__.__name__)
        fit.write_string_array('fit_parameter_names',self.fit_names)
        fit.write_string_array('fit_parameter_latex',self.fit_latex)
        fit.write_array('fit_boundary_low',np.array([x[0] for x in self.fit_boundaries]))
        fit.write_array('fit_boundary_high',np.array([x[1] for x in self.fit_boundaries]))

        ### This is the last sampled value ... should not be recorded to avoid confusion.
        #fit.write_list('fit_parameter_values',self.fit_values)
        #fit.write_list('fit_parameter_values_nomode',self.fit_values_nomode)
        return output

    def generate_profiles(self,solution,binning):
        from taurex.util.math import OnlineVariance
        from taurex import mpi
        """Generates sigma plots for profiles"""
        from taurex.util.util import weighted_avg_and_std
        weights = []
        tp_profiles = OnlineVariance()
        active_gases = OnlineVariance()
        inactive_gases = OnlineVariance()
        #tau_profile = OnlineVariance()
        binned_spectrum = OnlineVariance()
        native_spectrum = OnlineVariance()



        sample_list = None
        if mpi.get_rank() == 0:
            sample_list = list(self.sample_parameters(solution))
        
        sample_list = mpi.broadcast(sample_list)

        self.debug('We all got %s',sample_list)

        self.info('------------Profile generation step------------------')

        self.info('We are sampling %s points for the profiles',len(sample_list))

        rank = mpi.get_rank()
        size = mpi.nprocs()


        self.info('I will only iterate through partitioned %s points (the rest is in parallel)',len(sample_list)//size)
        disableLogging()
        count= 0

        for parameters,weight in sample_list[rank::size]: #sample likelihood space and get their parameters
            self.update_model(parameters)

            if rank ==0 and count % 10 ==0 and count >0:
                self.error('Progress {}%'.format(count*100.0        /(len(sample_list)/size)))

            count +=1
            weights.append(weight)
            native_grid,native,tau,_ = self._model.model(wngrid=binning,cutoff_grid=False)
            binned = self._binner.bindown(native_grid,native)[1]
            #tau_profile.update(tau,weight=weight)
            tp_profiles.update(self._model.temperatureProfile,weight=weight)
            active_gases.update(self._model.chemistry.activeGasMixProfile,weight=weight)
            inactive_gases.update(self._model.chemistry.inactiveGasMixProfile,weight=weight)
            binned_spectrum.update(binned,weight=weight)
            native_spectrum.update(native,weight=weight)
        enableLogging()


        total_counts = sum(all_gather(count))

        if total_counts > 0:
            tp_std = np.sqrt(tp_profiles.parallelVariance())
            active_std = np.sqrt(active_gases.parallelVariance())
            inactive_std = np.sqrt(inactive_gases.parallelVariance())

            #tau_std = np.sqrt(tau_profile.parallelVariance())

            binned_std = np.sqrt(binned_spectrum.parallelVariance())
            native_std = np.sqrt(native_spectrum.parallelVariance())
        else:
            self.warning('WEIGHTS ARE ALL ZERO, SETTING PROFILES STD TO ZERO')
            tp_std = np.zeros_like(tp_profiles)
            active_std = np.zeros_like(active_gases)
            inactive_std = np.zeros_like(inactive_gases)

           # tau_std = np.zeros_like(tau_profile)

            binned_std = np.zeros_like(binned_spectrum)
            native_std = np.zeros_like(native_spectrum)
        tau_std = None
        return tp_std,active_std,inactive_std,tau_std,binned_std,native_std

    def generate_solution(self,output_size=OutputSize.heavy):
        from taurex.util.output import generate_profile_dict,generate_spectra_dict,store_contributions
        """Generates a dictionar with all solutions and other useful parameters"""
        solution_dict = {}
        self.info('Generating spectra and profiles')
        #Loop through each solution, grab optimized parameters and anything else we want to store 
        for solution,optimized,values in self.get_solution(): 
            
            self.info('Computing solution %s',solution)
            sol_values = {}
            #print(values)
            #Include extra stuff we might want to store (provided by the child)
            for k,v in values:
                sol_values[k] = v
            
            self.update_model(optimized) #Update the model with optimized values

  

            sol_values['Profiles']=generate_profile_dict(self._model)
            opt_result = self._model.model(wngrid=self._observed.wavenumberGrid,cutoff_grid=False) #Run the model

            sol_values['Spectra'] = self._binner.generate_spectrum_output(opt_result,output_size=output_size)


            sol_values['Spectra']['Contributions'] = store_contributions(self._binner,self._model,output_size=output_size-3)


            #Store profiles here
            tp_std,active_std,inactive_std,tau_std,binned_std,native_std= self.generate_profiles(solution,self._observed.wavenumberGrid)
            

            sol_values['Spectra']['native_std'] = native_std
            sol_values['Spectra']['binned_std'] = binned_std
            sol_values['Profiles']['temp_profile_std']=tp_std
            sol_values['Profiles']['active_mix_profile_std']=active_std
            sol_values['Profiles']['inactive_mix_profile_std']=inactive_std






            solution_dict['solution{}'.format(solution)] = sol_values

        for solution,optimized,values in self.get_solution(): 
            mu = self.compute_mu_derived_trace(solution)
            solution_dict['solution{}'.format(solution)]['fit_params']['mu_derived'] = mu


        return solution_dict



    def compute_mu_derived_trace(self,solution):
        from taurex.util.util import quantile_corner
        from taurex.constants import AMU
        sigma_frac = self._sigma_fraction
        self._sigma_fraction = 1.0
        mu_trace = []
        weights = []
        for parameters,weight in self.sample_parameters(solution):
            self.update_model(parameters)   
            self._model.initialize_profiles()
            mu_trace.append(self._model.chemistry.muProfile[0]/AMU)
            weights.append(weight)



        self._sigma_fraction = sigma_frac
        q_16, q_50, q_84 = quantile_corner(np.array(mu_trace), [0.16, 0.5, 0.84],
                                           weights=np.array(weights))

        #print(mu_trace)
        mu_derived = {
            'value' : q_50,
            'sigma_m' : q_50-q_16,
            'sigma_p' : q_84-q_50,
            'trace': mu_trace,
        }
        return mu_derived

    def sample_parameters(self,solution):
        raise NotImplementedError



    def get_solution(self):
        raise NotImplementedError



    def write(self,output):
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
        









