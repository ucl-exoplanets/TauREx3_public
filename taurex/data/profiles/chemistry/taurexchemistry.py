from .chemistry import Chemistry
from taurex.data.fittable import fitparam
import numpy as np
import math

from taurex.util import *
class TaurexChemistry(Chemistry):


    def __init__(self,mode='absolute',ratio_molecule_index=0,n2_mix_ratio=0,he_h2_ratio=0.1764):
        super().__init__('ChemistryModel')

        self._mode = mode
        self._ratio_molecule=ratio_molecule_index

        self._n2_mix_ratio = n2_mix_ratio
        self._he_h2_mix_ratio = he_h2_ratio

        self._gases = []
        self.active_mixratio_profile = None
        self.inactive_mixratio_profile = None

    def addGas(self,gas):
        self._gases.append(gas)




    @property
    def ratioMoleculeProfile(self):
        if self._mode in ('Relative','relative',):
            return self._gases[self._ratio_molecule].mixRatioProfile
        else:
            return 1.0


    @property
    def activeGases(self):
        return [gas.molecule for gas in self._gases]


    @property
    def inActiveGases(self):
        return  ['H2', 'HE', 'N2']


    def fitting_parameters(self):
        """Overwrites the fitting paramter with ours"""
        full_dict = {}
        for gas in self._gases:
            full_dict.update(gas.fitting_parameters())
        
        full_dict.update(self._param_dict)
        

        return full_dict

    def initialize_chemistry(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        self.info('Initializing chemistry model')
        self.active_mixratio_profile = np.zeros(shape=(len(self._gases),nlayers))
        self.inactive_mixratio_profile = np.zeros((len(self.inActiveGases), nlayers))

        for idx,gas in enumerate(self._gases):
            gas.initialize_profile(nlayers,temperature_profile,pressure_profile,altitude_profile)
            self.active_mixratio_profile[idx,:] = gas.mixProfile


        


        #Since this can either be a scalar one or an array lets do it the old fashion way


        self.compute_absolute_gas_profile()
        

        super().initialize_chemistry(nlayers,temperature_profile,pressure_profile,altitude_profile)
        


    @property
    def activeGasMixProfile(self):
        return self.active_mixratio_profile

    @property
    def inActiveGasMixProfile(self):
        return self.inactive_mixratio_profile


    def compute_absolute_gas_profile(self):

        
        self.inactive_mixratio_profile[2, :] = self._n2_mix_ratio
        # first get the sum of the mixing ratio of all active gases


        active_mixratio_sum = np.sum(self.active_mixratio_profile, axis = 0)
        
        active_mixratio_sum += self.inactive_mixratio_profile[2, :]
        


        mixratio_remainder = 1. - active_mixratio_sum
        self.inactive_mixratio_profile[0, :] = mixratio_remainder/(1. + self._he_h2_mix_ratio) # H2
        self.inactive_mixratio_profile[1, :] =  self._he_h2_mix_ratio * self.inactive_mixratio_profile[0, :] 



    @fitparam(param_name='N2',param_latex=molecule_texlabel('N2'),default_mode='log',default_fit=False,default_bounds=[1e-12,1.0])
    def N2MixRatio(self):
        return self._n2_mix_ratio
    
    @N2MixRatio.setter
    def N2MixRatio(self,value):
        self._n2_mix_ratio = value

    @fitparam(param_name='H2_He',param_latex=molecule_texlabel('H$_2$/He'),default_mode='log',default_fit=False,default_bounds=[1e-12,1.0])
    def H2HeMixRatio(self):
        return self._he_h2_mix_ratio
    
    @H2HeMixRatio.setter
    def H2HeMixRatio(self,value):
        self._he_h2_mix_ratio = value


    def write(self,output):
        gas_entry = super().write(output)
        gas_entry.write_scalar('n2_mix_ratio',self._n2_mix_ratio)
        gas_entry.write_scalar('he_h2_ratio',self._he_h2_mix_ratio)
        for gas in self._gases:
            gas.write(gas_entry)

        return gas_entry

        