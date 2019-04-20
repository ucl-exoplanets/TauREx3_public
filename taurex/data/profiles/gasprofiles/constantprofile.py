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


    def __init__(self,active_gases,active_gas_mix_ratio,n2_mix_ratio=0,he_h2_ratio=0.17647,mode='linear'):
        super().__init__('ConstantGasProfile',active_gases,active_gas_mix_ratio,n2_mix_ratio,he_h2_ratio,mode=mode)

        self.setup_fitting_params()
    
    def add_active_gas_param(self,idx):
        mol_name = self.active_gases[idx]

        param_name = mol_name
        param_tex = molecule_texlabel(mol_name)
        if self.isInLogMode:
            param_name = 'log_{}'.format(mol_name)
            param_tex = 'log({})'.format(molecule_texlabel(mol_name))
        
        def read_mol():
            return self.readableValue(self.active_gas_mix_ratio[idx])
        def write_mol(value):
            self.active_gas_mix_ratio[idx] = self.writeableValue(value)

        fget = read_mol
        fset = write_mol
        
        bounds = [1.0e-12, 0.1]
        if self.isInLogMode:
            bounds=[-12,-1]
        
        default_fit = False
        self.add_fittable_param(param_name,param_tex,fget,fset,default_fit,bounds)
    def setup_fitting_params(self):
        
        for idx,mol_name in enumerate(self.active_gases):
            self.add_active_gas_param(idx)




            





            