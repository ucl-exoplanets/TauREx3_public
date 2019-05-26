from taurex.log import Logger
from taurex.data.fittable import Fittable,fitparam
from taurex.output.writeable import Writeable

class Gas(Fittable,Logger,Writeable):

    def __init__(self,name,molecule_name):
        Logger.__init__(self,name)
        Fittable.__init__(self)
        self._molecule_name = molecule_name.upper()
        self.mix_profile=None
    

    @property
    def molecule(self):
        return self._molecule_name

    @property
    def mixProfile(self):
        raise NotImplementedError

    def initialize_profile(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        pass

    def write(self,output):

        gas_entry = output.create_group(self.molecule)
        return gas_entry