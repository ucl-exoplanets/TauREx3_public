from .tpprofile import TemperatureProfile
import numpy as np


class Guillot2010(TemperatureProfile):
    """

    TP profile from Guillot 2010, A&A, 520, A27 (equation 49)
    Using modified 2stream approx. from Line et al. 2012, ApJ, 749,93 (equation 19)

    TODO: ACTUALLY IMPLEMENT IT MATE

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

