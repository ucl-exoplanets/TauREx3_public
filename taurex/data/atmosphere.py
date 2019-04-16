from taurex.log import Logger
import numpy as np


class Atmosphere(Logger):
    """This class defines the atmosphere

    """



    def __init__(self,star, planet,nlayers,
                        t_profile,
                        p_profile,
                        dens_profile,
                        
                        gas_profile):
        super().__init__('Atmosphere')

        self._star = star
        self._planet = planet
        self._t_profile = t_profile
        self._gas_profile = gas_profile
        self.setup_pressure_profile(nlayers)
    

    def setup_pressure_profile(self,nlayers):
        """Sets up the pressure profile for the atmosphere model
        
        Parameters
        ----------
        
        
        """
        self._nlayers = nlayers

        # set pressure profile of layer boundaries
        press_exp = np.linspace(np.log(self.params.atm_min_pres), np.log(self.params.atm_max_pres), self.nlevels)
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
    def tpProfile(self):
        return self._tp_profile


    def setStar(self,star):
        pass
    
    def setPlanet(self,planet):
        pass

