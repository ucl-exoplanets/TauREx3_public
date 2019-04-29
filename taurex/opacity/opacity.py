from taurex.log import Logger


class Opacity(Logger):
    """
    This is the base class for computing opactities

    """
    
    def __init__(self,name):
        super().__init__(name)

    
    @property
    def moleculeName(self):
        raise NotImplementedError

    @property
    def wavenumberGrid(self):
        raise NotImplementedError

    @property
    def temperatureGrid(self):
        raise NotImplementedError
    
    @property
    def pressureGrid(self):
        raise NotImplementedError


    def opacity(self,temperature,pressure):
        raise NotImplementedError
