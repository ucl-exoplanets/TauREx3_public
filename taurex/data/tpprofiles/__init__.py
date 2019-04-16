from taurex.log import Logger


class TPProfile(Logger):
    """
    Defines temperature pressure profile for an atmosphere

    Must define a profile() function that returns
    a Temperature, Pressure, Column Density (T, P, X) grid 

    """
    

    def __init__(self):
        pass

    

    def profile(self):
        raise NotImplementedError