from .tprofile import TemperatureProfile
import numpy as np


class TemperatureArray(TemperatureProfile):
    """

    Temperature profile loaded from array

    """

    def __init__(self, tp_array=[2000, 1000], p_points=None):
        super().__init__(self.__class__.__name__)

        self._tp_profile = np.array(tp_array)
        if p_points is not None:
            self._p_profile = np.array(p_points)
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
        else:
            interp_temp = np.log10(self._p_profile)
            interp_array = np.log10(self.pressure_profile)

        return np.interp(interp_array[::-1], interp_temp[::-1], self._tp_profile[::-1])[::-1]

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
