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

    def cia(self,temperature):
        raise NotImplementedError