from taurex.log import Logger


class TPProfile(Logger):
    """
    Defines temperature pressure profile for an atmosphere

    Must define a profile() function that returns
    a Temperature, Pressure, Column Density (T, P, X) grid 

    """
    

    def __init__(self,name,nlayers,pressure_profile):
        super().__init__(name)
        self.nlayers=nlayers
        self.nlevels = nlayers+1
        self.pressure_profile = pressure_profile

    

    def profile(self):
        raise NotImplementedError