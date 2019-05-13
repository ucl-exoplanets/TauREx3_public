from taurex.log import Logger
from taurex.data.fittable import fitparam,Fittable
from taurex.output.writeable import Writeable
class TemperatureProfile(Fittable,Logger,Writeable):
    """
    Defines temperature profile for an atmosphere

    Must define a profile() function that returns
    a Temperature, (T, P, X) grid 

    """
    

    def __init__(self,name):
        Logger.__init__(self,name)
        Fittable.__init__(self)


    def initialize_profile(self,planet,nlayers,pressure_profile):
        self.nlayers=nlayers
        self.nlevels = nlayers+1
        self.pressure_profile = pressure_profile
        self.planet = planet


    @property
    def profile(self):
        raise NotImplementedError


    def write(self,output):
        temperature = output.create_group('Temperature')
        temperature.write_string('temperature_type',self.__class__.__name__)
        temperature.write_array('profile',self.profile)
        return temperature

    