from .tprofile import TemperatureProfile
import numpy as np


class Isothermal(TemperatureProfile):
    """An isothermal temperature-pressure profile

    Parameters
    ----------
    
    iso_temp : :obj:`float`
        Isothermal Temperature to set

    """


    def __init__(self,iso_temp):
        super().__init__('Isothermal')

        self._iso_temp = iso_temp
    
    def profile(self):
        """Returns an isothermal temperature profile

        Returns: :obj:np.array(float)
            temperature profile
        """

        T = np.zeros((self.nlayers))
        T[:] = self._iso_temp


        return T

