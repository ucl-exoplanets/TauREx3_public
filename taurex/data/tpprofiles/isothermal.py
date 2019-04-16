from .tpprofile import TPProfile
import numpy as np


class Isothermal(TPProfile):
    """An isothermal temperature-pressure profile

    Parameters
    ----------
    n_layers : :obj:`int`
        Number of defined layers in the profile
    
    temp : :obj:`float`
        Temperature to set

    """


    def __init__(self,nlayers,pressure_profile,temp):
        super().__init__('Isothermal',nlayers,pressure_profile)

        
        self._tp = np.zeros(shape=(self.nlayers,))

        self._tp[...] = temp

    
    def profile(self):
        """Returns an isothermal temperature-pressure profile

        Returns: :obj:np.array(float)
            temperature profile
        """

        return self._tp

