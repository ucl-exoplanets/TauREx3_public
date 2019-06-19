from .gas import Gas
from taurex.util import molecule_texlabel
class ConstantGas(Gas):
    """

    Constant gas profile. Mixing profile is the same at each layer of the atmosphere




    Parameters
    -----------
    molecule_name : str
        Name of molecule

    mix_ratio : float
        Mixing ratio of the molecule

    """


    def __init__(self,molecule_name='H2O',mix_ratio=1e-5):
        super().__init__('ConstantGas',molecule_name)
        self._mix_ratio = mix_ratio
        self.add_active_gas_param()

    @property
    def mixProfile(self):
        """

        Returns
        -------
        mix_ratio : float
            Mix ratio for every layer

        """

        return self._mix_ratio

    def add_active_gas_param(self):
        """
        Adds the mixing ratio as a fitting parameter
        as the name of the molecule
        """
        
        mol_name = self.molecule
        param_name = self.molecule
        param_tex = molecule_texlabel(mol_name)
        
        def read_mol(self):
            return self._mix_ratio
        def write_mol(self,value):
            self._mix_ratio = value

        fget = read_mol
        fset = write_mol
        
        bounds = [1.0e-12, 0.1]
        
        default_fit = False
        self.add_fittable_param(param_name,param_tex,fget,fset,'log',default_fit,bounds) 
    

    def write(self,output):
        gas_entry = super().write(output)
        gas_entry.write_scalar('mix_ratio',self.mixProfile)

        return gas_entry
