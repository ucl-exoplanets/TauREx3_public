from .gasprofile import TaurexGasProfile
import numpy as np
from taurex.data.fittable import fitparam
from taurex.util import molecule_texlabel
class ConstantGasProfile(TaurexGasProfile):
    """

    Constant gas profile


    Parameters
    -----------


    """


    def __init__(self,active_gases=['H2O','CH4'],active_gas_mix_ratio=[1e-5,1e-6],n2_mix_ratio=0,he_h2_ratio=0.17647):
        super().__init__('ConstantGasProfile',active_gases,active_gas_mix_ratio,n2_mix_ratio,he_h2_ratio)

        self.setup_fitting_params()
    

    def setup_fitting_params(self):
        
        for idx,mol_name in enumerate(self.active_gases):
            self.add_active_gas_param(idx)




            





            