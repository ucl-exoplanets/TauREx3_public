from .pressureprofile import PressureProfile 
import numpy as np


class ArrayPressureProfile(PressureProfile):

    def __init__(self, array, reverse=False):
        
        super().__init__(self.__class__.__name__, array.shape[-1])
        if reverse:
            self.pressure_profile = array[::-1]
        else:
            self.pressure_profile = array

    def compute_pressure_profile(self):
        """
        Sets up the pressure profile for the atmosphere model

        """

        logp = np.log10(self.pressure_profile)
        gradp = np.gradient(logp)

        self.pressure_profile_levels = \
            10**np.append(logp-gradp/2, logp[-1]+gradp[-1]/2)

    @property
    def profile(self):
        return self.pressure_profile

    def write(self, output):
        pressure = super().write(output)

        return pressure

    @classmethod
    def input_keywords(self):
        return ['array', 'fromarray',]