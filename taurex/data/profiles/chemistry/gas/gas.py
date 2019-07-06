from taurex.log import Logger
from taurex.data.fittable import Fittable,fitparam
from taurex.output.writeable import Writeable

class Gas(Fittable,Logger,Writeable):
    """
    This class is a base for a single molecule or gas.
    Its used to describe how it mixes at each layer and combined
    with :class:`~taurex.data.profile.chemistry.taurexchemistry.TaurexChemistry`
    is used to build a chemical profile of the planets atmosphere.

    Parameters
    -----------

    name :str
        Name used in logging
    
    molecule_name : str
        Name of molecule
    



    """


    def __init__(self,name,molecule_name):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        self._molecule_name = molecule_name
        self.mix_profile=None
    

    @property
    def molecule(self):
        """
        Returns
        -------
        molecule_name : str
            Name of molecule
        """
        return self._molecule_name

    @property
    def mixProfile(self):
        """
        **Unimplemented**

        Should return mix profile of molecule/gas at each layer

        """
        raise NotImplementedError

    def initialize_profile(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        """
        Override to initialize and compute mix profile
        """
        pass

    def write(self,output):

        gas_entry = output.create_group(self.molecule)
        gas_entry.write_string('gas_type',self.__class__.__name__)
        return gas_entry