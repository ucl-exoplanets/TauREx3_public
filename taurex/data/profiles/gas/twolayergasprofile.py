from .gasprofile import ComplexGasProfile
import numpy as np
from taurex.data.fittable import fitparam
from taurex.util import molecule_texlabel,movingaverage



class TwoLayerGasProfile(ComplexGasProfile):

    def __init__(self,
                active_gases=['H2O','CH4'],
                active_gas_mix_ratio=[1e-5,1e-6],
                active_complex_gases=['CH4'],
                active_gases_mixratios_surface=[1e-4],
                active_gases_mixratios_top=[1e-8],
                active_gases_mixratios_P = [1e3],
                active_gases_smooth=10,
                n2_mix_ratio=0,he_h2_ratio=0.17647,mode='linear'):

        super().__init__('2-point gas',active_gases,
                active_gas_mix_ratio,
                active_complex_gases,
                active_gases_mixratios_surface,
                active_gases_mixratios_top,
                n2_mix_ratio,he_h2_ratio,mode)
        self.active_gases_mixratios_P = active_gases_mixratios_P 
        self.active_complex_gases_smooth = active_gases_smooth

        self.add_P_param()
    def add_P_param(self):
        for idx,mol_name in enumerate(self.active_complex_gases):
            mol_tex = molecule_texlabel(mol_name)

            param_P = 'P {}'.format(mol_name)
            param_P_tex = '{}'.format(mol_tex)

            def read_P(self,idx=idx):
                return self.active_gases_mixratios_P[idx]
            def write_P(self,value,idx=idx):
                self.active_gases_mixratios_P[idx] = value

            fget_P = read_P
            fset_P = write_P

            bounds = [1.0e-12, 0.1]

            default_fit = False
            self.add_fittable_param(param_P,param_P_tex ,fget_P,fset_P,default_fit,bounds)      

    def compute_complex_gas_profile(self):

        for i,j in self.complex_gas_iter():
            smooth_window = self.active_complex_gases_smooth
            self.P_layer = np.abs(self.pressure_profile - self.active_gases_mixratios_P[j]).argmin()

            Pnodes = [self.pressure_profile[0], self.pressure_profile[int(self.P_layer-smooth_window/2)], self.pressure_profile[int(self.P_layer+smooth_window/2)], self.pressure_profile[-1]]
            Cnodes = [self.active_gases_mixratios_surface[j], self.active_gases_mixratios_surface[j], self.active_gases_mixratios_top[j],self.active_gases_mixratios_top[j]]

            chemprofile = self.active_mixratio_profile[i, :]
            chemprofile = 10**np.interp((np.log(self.pressure_profile[::-1])), np.log(Pnodes[::-1]), np.log10(Cnodes[::-1]))


            wsize = self.nlayers * (smooth_window / 100.0)
            if (wsize % 2 == 0):
                wsize += 1
            C_smooth = 10**movingaverage(np.log10(chemprofile), int(wsize))
            border = np.int((len(chemprofile) - len(C_smooth)) / 2)
            foo = chemprofile[::-1]
            foo[border:-border] = C_smooth[::-1]
            self.active_mixratio_profile[i, :] = foo[:]
    

    def write(self,output):

        gas = super().write(output)

        gas.write('active_gases_mixratios_P',self.active_gases_mixratios_P)
        gas.write('active_gases_smooth',self.active_complex_gases_smooth)

        return gas