from .tprofile import TemperatureProfile
import numpy as np
from scipy.interpolate import interp1d

class TemperatureArray(TemperatureProfile):
    """

    Temperature profile loaded from array

    """

    def __init__(self, tp_array=[2000, 1000], p_points=None, reverse=False):
        super().__init__(self.__class__.__name__)

        self._tp_profile = np.array(tp_array)
        if reverse:
            self._tp_profile = self._tp_profile[::-1]
        if p_points is not None:
        
            self._p_profile = np.array(p_points)
            if reverse:
                self._p_profile = self._p_profile[::-1]
            self._func = interp1d(np.log10(self._p_profile), self._tp_profile,
                                  bounds_error=False,
                                  fill_value=(self._tp_profile[-1],
                                  self._tp_profile[0]))
        else:
            self._p_profile = None
        
    @property
    def profile(self):
        """Returns an isothermal temperature profile

        Returns: :obj:np.array(float)
            temperature profile
        """

        # if self._tp_profile.shape[0] == self.nlayers:
        #     return self._tp_profile
        # else:
        if self._p_profile is None:
            if self._tp_profile.shape[0] == self.nlayers:
                return self._tp_profile
            interp_temp = np.linspace(1.0, 0.0, self._tp_profile.shape[0])
            interp_array = np.linspace(1.0, 0.0, self.nlayers)
            return np.interp(interp_array[::-1], interp_temp[::-1],
                            self._tp_profile[::-1])
        else:
            #print(self._p_profile)
            interp_array = np.log10(self.pressure_profile)
            return self._func(interp_array)


    def write(self, output):

        temperature = super().write(output)

        temperature.write_scalar('tp_array', self._tp_profile)

        return temperature

    @classmethod
    def input_keywords(cls):
        """
        Return all input keywords
        """
        return ['array', 'fromarray', ]
