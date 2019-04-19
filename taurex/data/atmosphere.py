from taurex.log import Logger
import numpy as np
from .fittable import fitparam,Fittable
from taurex.constants import KBOLTZ

class Atmosphere(Fittable,Logger):
    """This class defines the atmosphere

    """



    def __init__(self,star, 
                    planet,
                    nlayers=100,
                    atm_min_pressure=1e-4,
                    atm_max_pressure=1e6,
                    dens_profile=None,
                    t_profile=None,
                    gas_profile=None):
        super().__init__('Atmosphere')

        self._star = star
        self._planet = planet
        self._t_profile = t_profile
        self._gas_profile = gas_profile
        
        self._atm_min_pressure = atm_min_pressure
        
        self._atm_max_pressure = atm_max_pressure

        self.setup_pressure_profile(nlayers)


    def setup_pressure_profile(self,nlayers):
        """Sets up the pressure profile for the atmosphere model
        
        Parameters
        ----------
        
        
        """
        self._nlayers = nlayers

        # set pressure profile of layer boundaries
        press_exp = np.linspace(np.log(self._atm_min_pressure), np.log(self._atm_max_pressure), self.nLevels)
        self.pressure_profile_levels =  np.exp(press_exp)[::-1]

        # get mid point pressure between levels (i.e. get layer pressure) computing geometric
        # average between pressure at n and n+1 level
        self.pressure_profile = np.power(10, np.log10(self.pressure_profile_levels)[:-1]+
                                         np.diff(np.log10(self.pressure_profile_levels))/2.)


    @property
    def nLevels(self):
        return self._nlayers+1
    

    @property
    def star(self):
        return self._star
    
    @property
    def planet(self):
        return self._planet

    
    @property
    def temperatureProfile(self):
        return self._t_profile.profile()

    @property
    def pressureProfile(self):
        return self.pressure_profile

    @property
    def densityProfile(self):
        return (self.pressureProfile)/(KBOLTZ*self.temperatureProfile)




