from taurex.log import Logger
from taurex.data.fittable import fitparam,Fittable
import numpy as np
class GasProfile(Fittable,Logger):
    """
    Defines gas profiles

    """

    inactive_gases = ['H2', 'HE', 'N2']


    def __init__(self,name,active_gases,active_gas_mix_ratio,n2_mix_ratio=0,he_h2_ratio=0.17647):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        self._active_gases = active_gases
        self._active_gas_mix_ratio = active_gas_mix_ratio
        self._n2_mix_ratio = n2_mix_ratio
        self._he_h2_mix_ratio = he_h2_ratio

    def initialize_profile(self,nlayers,pressure_profile):
        self.nlayers=nlayers
        self.nlevels = nlayers+1
        self.pressure_profile = pressure_profile

        self.active_mixratio_profile=np.zeros((len(self._active_gases), self.nlayers))

        self.compute_active_gas_profile()
        self.compute_inactive_gas_profile()

    def compute_active_gas_profile(self):

        for idx,ratio in self._active_gas_mix_ratio:
            self.active_mixratio_profile[idx, :] = ratio
    
    def compute_inactive_gas_profile(self):

        self.inactive_mixratio_profile = np.zeros((len(self.inactive_gases), self.nlayers))
        self.inactive_mixratio_profile[2, :] = self._n2_mix_ratio
        # first get the sum of the mixing ratio of all active gases
        if len(self._active_gases) > 1:
            active_mixratio_sum = np.sum(self.active_mixratio_profile, axis = 0)
        else:
            active_mixratio_sum = np.copy(self.active_mixratio_profile)
        
        active_mixratio_sum += self.inactive_mixratio_profile[2, :]