from .gasprofile import ComplexGasProfile
import numpy as np
from taurex.data.fittable import fitparam
from taurex.util import molecule_texlabel



class TwoPointGasProfile(ComplexGasProfile):

    def __init__(self,
                active_gases,
                active_gas_mix_ratio,
                active_complex_gases,
                active_gases_mixratios_surface,
                active_gases_mixratios_top,
                n2_mix_ratio=0,he_h2_ratio=0.17647,mode='linear'):

        super().__init__('2-point gas',active_gases,
                active_gas_mix_ratio,
                active_complex_gases,
                active_gases_mixratios_surface,
                active_gases_mixratios_top,
                n2_mix_ratio,he_h2_ratio,mode)
    

    def compute_complex_gas_profile(self):

        for i,j in self.complex_gas_iter():
            chem_surf = self.active_gases_mixratios_surface[j]
            chem_top = self.active_gases_mixratios_top[j]
            p_surf = self.pressure_profile[0]
            p_top = self.pressure_profile[-1]

            a = (np.log10(chem_surf)-np.log10(chem_top))/(np.log10(p_surf)-np.log10(p_top))
            b = np.log10(chem_surf)-a*np.log10(p_surf)

            self.active_mixratio_profile[i, 1:-1] = pow(10, a * np.log10(self.pressure_profile[1:-1])+b)
            self.active_mixratio_profile[i, 0] = chem_surf
            self.active_mixratio_profile[i, -1] = chem_top