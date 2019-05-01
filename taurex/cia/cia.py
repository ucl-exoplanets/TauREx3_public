from taurex.log import Logger


class CIA(Logger):
    """
    This is the base class for collisionally induced absorption opacities

    """
    
    def __init__(self,name,pair_name):
        super().__init__(name)

        self._pair_name = pair_name

    
    @property
    def pairName(self):
        return self._pair_name


    @property
    def pairOne(self):
        return self._pair_name.split('-')[0]

    @property
    def pairTwo(self):
        return self._pair_name.split('-')[-1]

    @property
    def wavenumberGrid(self):
        raise NotImplementedError

    @property
    def temperatureGrid(self):
        raise NotImplementedError

    def cia(self,temperature):
        raise NotImplementedError