from .gasprofile import GasProfile
from taurex.external.ace import md_ace
from taurex.data.fittable import fitparam
import numpy as np
import math
class ACEGasProfile(GasProfile):

    ace_H_solar = 12.
    ace_He_solar = 10.93
    ace_C_solar = 8.43
    ace_O_solar = 8.69
    ace_N_solar = 7.83 


    def __init__(self,active_gases,metallicity=1,co_ratio=0.54951,therm_file = None,spec_file=None,mode='linear'):
        super().__init__('ACE',mode=mode)

        self.inactive_gases = ['H2', 'HE', 'N2']
        self.ace_metallicity = metallicity
        self.ace_co = co_ratio
        self.set_ace_params()
        self.active_gases = active_gases
        self._get_files(therm_file,spec_file)
        self._get_gas_mask()


    def _get_files(self,therm_file,spec_file):
        import os
        import taurex.external

        path_to_files = os.path.join(os.path.abspath(os.path.dirname(taurex.external.__file__)),'ACE')
        self._specfile = spec_file
        self._thermfile = therm_file
        if self._specfile is None:
            self._specfile = os.path.join(path_to_files,'composes.dat')
        if self._thermfile is None:
            self._thermfile = os.path.join(path_to_files,'NASA.therm')
    
    def _get_gas_mask(self):

        self._active_mask = np.ndarray(shape=(105,),dtype=np.bool)
        self._inactive_mask = np.ndarray(shape=(105,),dtype=np.bool)

        self._active_mask[...] = False
        self._inactive_mask[...] = False

        with open(self._specfile, 'r') as textfile:
            for line in textfile:
                sl = line.split()
                idx = int(sl[0])
                molecule = sl[1].upper()
                if molecule in self.active_gases:
                    self._active_mask[idx] = True
                if molecule in self.inactive_gases:
                    self._inactive_mask[idx] = True
        
    
    def set_ace_params(self):

        # set O, C and N abundances given metallicity (in solar units) and CO ratio
        self.O_abund_dex = math.log10(self.ace_metallicity * (10**(self.ace_O_solar-12.)))+12.
        self.N_abund_dex = math.log10(self.ace_metallicity * (10**(self.ace_N_solar-12.)))+12.
        self.C_abund_dex = self.O_abund_dex + math.log10(self.ace_co)

        # H and He don't change
        self.H_abund_dex = self.ace_H_solar
        self.He_abund_dex = self.ace_He_solar


    def compute_active_gas_profile(self):
        self._ace_profile = md_ace(self._specfile,self._thermfile,self.altitude_profile/1000.0,self.pressure_profile/1.e5,self.temperature_profile,
            self.He_abund_dex,self.C_abund_dex,self.O_abund_dex,self.N_abund_dex)
        
        self.active_mixratio_profile= self._ace_profile[self._active_mask,:]
        self.inactive_mixratio_profile = self._ace_profile[self._inactive_mask,:]

    
    def compute_inactive_gas_profile(self):
        pass
    

    @fitparam(param_name='ace_log_metallicity',param_latex='log(Metallicity)',default_fit=False,default_bounds=[ -1, 4])
    def aceMetallicity(self):
        return math.log10(self.ace_metallicity)
    
    @aceMetallicity.setter
    def aceMetallicity(self,value):
        self.ace_metallicity = math.pow(10.0,value)
    

    @fitparam(param_name='ace_co',param_latex='C/O',default_fit=False,default_bounds=[0, 2])
    def aceCORatio(self):
        return self.ace_co
    
    @aceCORatio.setter
    def aceCORatio(self,value):
        self.ace_co = value