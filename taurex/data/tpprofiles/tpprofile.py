from taurex.log import Logger


class TemperatureProfile(Logger):
    """
    Defines temperature profile for an atmosphere

    Must define a profile() function that returns
    a Temperature, (T, P, X) grid 

    """
    

    def __init__(self,name):
        super().__init__(name)



    def initialize_profile(self,planet,nlayers,pressure_profile):
        self.nlayers=nlayers
        self.nlevels = nlayers+1
        self.pressure_profile = pressure_profile
        self._planet = planet



    def profile(self):
        raise NotImplementedError