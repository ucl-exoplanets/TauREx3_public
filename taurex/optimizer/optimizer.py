from taurex.log import Logger
import numpy as np


class Optimizer(Logger):

    def __init__(self,name,observed=None,model=None,wngrid=None):
        super().__init__(name)

        self._model = model
        self._observed = observed
        self._wngrid = None
        self._model_callback = None

    def set_model(self,model):
        self._model = model

    def set_observed(self,observed):
        self._observed = observed

    def set_wavenumber_grid(self,wngrid):
        self._wngrid = wngrid

    def compile_params(self):
        self.info('Initializing parameters')
        self.fitting_parameters=[]
        # param_name,param_latex,
        #                 fget.__get__(self),fset.__get__(self),
        #                         default_fit,default_bounds
        for params in self._model.fittingParameters.values():
            name,latex,fget,fset,to_fit,bounds = params

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
            name,latex,fget,fset,to_fit,bounds = params
            self.info('{}: Value: {} Boundaries:{}'.format(name,fget(),bounds))


    def update_model(self,fit_params):

        for value,param in zip(fit_params,self.fitting_parameters):
            name,latex,fget,fset,to_fit,bounds = param
            fset(value)


    @property
    def fit_values(self):
        return [c[2]() for c in self.fitting_parameters]

    @property
    def fit_boundaries(self):
        return [c[-1] for c in self.fitting_parameters]


    @property
    def fit_names(self):
        return [c[0] for c in self.fitting_parameters]


    def enable_fit(self,parameter):
        name,latex,fget,fset,to_fit,bounds = self._model.fittingParameters[parameter]
        to_fit = True
        self._model.fittingParameters[parameter]= (name,latex,fget,fset,to_fit,bounds)

    def disable_fit(self,parameter):
        name,latex,fget,fset,to_fit,bounds = self._model.fittingParameters[parameter]
        to_fit = False
        self._model.fittingParameters[parameter]= (name,latex,fget,fset,to_fit,bounds)
    
    def set_boundary(self,parameter,new_boundaries):
        name,latex,fget,fset,to_fit,bounds = self._model.fittingParameters[parameter]
        bounds = new_boundaries
        self._model.fittingParameters[parameter]= (name,latex,fget,fset,to_fit,bounds)

    def chisq_trans(self, fit_params,data,datastd):
        from taurex.util import bindown

        self.update_model(fit_params)

        obs_bins= self._observed.wavenumberGrid
        wngrid = self._wngrid

        #wngrid = obs_bins

        model_out,_,_ = self._model.model(wngrid)

        final_model =bindown(wngrid,model_out,obs_bins)
        res = (data[:-1] - final_model) / datastd[:-1]

        res = np.nansum(res*res)
        if res == 0:
            res = np.nan
        return res


    def compute_fit(self):
        raise NotImplementedError


    def fit(self):

        self.compile_params()

        self.compute_fit()








