from taurex.log import Logger
from taurex.util import get_molecular_weight,molecule_texlabel
from taurex.data.fittable import fitparam,Fittable
import numpy as np
from taurex.output.writeable import Writeable
import math

class Chemistry(Fittable,Logger,Writeable):
    """
    Defines Chemistry models

    """

    


    def __init__(self,name):
        Logger.__init__(self,name)
        Fittable.__init__(self)

    

    @property
    def activeGases(self):
        raise NotImplementedError
    

    @property
    def inActiveGases(self):
        raise NotImplementedError
    

    def initialize_chemistry(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        


        self.compute_mu_profile(nlayers)


    @property
    def activeGasMixProfile(self):
        raise NotImplementedError

    @property
    def inActiveGasMixProfile(self):
        raise NotImplementedError


    @property
    def muProfile(self):
        return self.mu_profile


    def get_gas_mix_profile(self,gas_name):
        if gas_name in self.activeGases:
            idx = self.activeGases.index(gas_name)
            return self.activeGasMixProfile[idx,:]
        elif gas_name in self.inActiveGases:
            idx = self.inActiveGases.index(gas_name)
            return self.inActiveGasMixProfile[idx,:]  
        else:
            raise KeyError  


    def compute_mu_profile(self,nlayers):
        self.mu_profile= np.zeros(shape=(nlayers,))
        if self.activeGasMixProfile is not None:
            for idx, gasname in enumerate(self.activeGases):
                self.mu_profile += self.activeGasMixProfile[idx,:]*get_molecular_weight(gasname)
        if self.inActiveGasMixProfile is not None:
            for idx, gasname in enumerate(self.inActiveGases):
                self.mu_profile += self.inActiveGasMixProfile[idx,:]*get_molecular_weight(gasname)


    def write(self,output):

        gas_entry = output.create_group('Chemistry')
        gas_entry.write_string('chemistry_type',self.__class__.__name__)
        gas_entry.write_string_array('active_gases',self.activeGases)
        gas_entry.write_string_array('inactive_gases',self.inActiveGases)
        gas_entry.write_array('active_gas_mix_profile',self.activeGasMixProfile)
        gas_entry.write_array('inactive_gas_mix_profile',self.inActiveGasMixProfile)
        gas_entry.write_array('mu_profile',self.muProfile)
        return gas_entry