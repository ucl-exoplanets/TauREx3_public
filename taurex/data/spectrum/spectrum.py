
from taurex.log import Logger



class BaseSpectrum(Logger):
    """A base class where spectrums are loaded (or later created)"""


    def __init__(self,name):
        super().__init__(name)

    


    @property
    def spectrumWL(self):
        raise NotImplementedError
    
    @property
    def spectrumWN(self):
        return 10000/self.spectrumWL

    @property
    def binEdges(self):
        raise NotImplementedError
    
    @property
    def binWidths(self):
        raise NotImplementedError



