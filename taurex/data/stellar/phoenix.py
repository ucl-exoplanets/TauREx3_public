from .star import BlackbodyStar


class PhoenixStar(BlackbodyStar):

    def __init__(self,temperature=5000,radius=1.0,phoenix_file_path=None):
        super().__init__(temperature=temperature,radius=radius)
        
    
    def preload_phoenix_spectra(self):
        pass
    
    def initialize(self,wngrid):
        if self.temperature > self._avail_max_temp or self.temperature < self._avail_min_temp:
            super().initialize(wngrid)
        else:
            pass