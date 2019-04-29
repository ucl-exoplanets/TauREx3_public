from taurex.log import Logger
from taurex.data.fittable import fitparam,Fittable
import numpy as np
class PressureProfile(Fittable,Logger):

    def __init__(self,name,nlayers):
        Fittable.__init__(self)
        Logger.__init__(self,name)


        self._nlayers = nlayers
        self._nlevels = self._nlayers + 1

    @property
    def nLayers(self):
        return self._nlayers
    
    @property
    def nLevels(self):
        return self._nlayers+1

    def compute_pressure_profile(self):
        raise NotImplementedError

    @property
    def profile(self):
        raise NotImplementedError


class SimplePressureProfile(PressureProfile):

    def __init__(self,nlayers=100,atm_min_pressure=1e-4,atm_max_pressure=1e6):
        super().__init__('pressure_profile',nlayers)
        self.pressure_profile = None
        self._atm_min_pressure = atm_min_pressure
        self._atm_max_pressure = atm_max_pressure

    def compute_pressure_profile(self):
        """Sets up the pressure profile for the atmosphere model
        
        
        """

        # set pressure profile of layer boundaries
        press_exp = np.linspace(np.log(self._atm_min_pressure), np.log(self._atm_max_pressure), self._nlevels)
        self.pressure_profile_levels =  np.exp(press_exp)[::-1]

        # get mid point pressure between levels (i.e. get layer pressure) computing geometric
        # average between pressure at n and n+1 level
        self.pressure_profile = np.power(10, np.log10(self.pressure_profile_levels)[:-1]+
                                         np.diff(np.log10(self.pressure_profile_levels))/2.)


    @fitparam(param_name='min_atm_pressure',param_latex='$P_\mathrm{min}$',default_fit=False,default_bounds=[0.1,1.0])
    def minAtmospherePressure(self):
        return self._atm_min_pressure
    
    @minAtmospherePressure.setter
    def minAtmospherePressure(self,value):
        self._atm_min_pressure = value

    @fitparam(param_name='max_atm_pressure',param_latex='$P_\mathrm{max}$',default_fit=False,default_bounds=[0.1,1.0])
    def maxAtmospherePressure(self):
        return self._atm_max_pressure
    
    @maxAtmospherePressure.setter
    def maxAtmospherePressure(self,value):
        self._atm_max_pressure = value


    @property
    def profile(self):
        return self.pressure_profile

    
