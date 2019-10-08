from taurex.log import Logger
import numpy as np

class Opacity(Logger):
    """
    This is the base class for computing opactities

    """
    
    def __init__(self,name):
        super().__init__(name)


    @property
    def resolution(self):
        raise NotImplementedError
    
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


    def compute_opacity(self,temperature,pressure,wngrid=None):
        raise NotImplementedError

    



    def opacity(self,temperature,pressure,wngrid=None):
        
        if wngrid is None:
            wngrid_filter = slice(None)
        else:
            wngrid_filter = (self.wavenumberGrid >= wngrid.min()) & (self.wavenumberGrid <= wngrid.max())

        orig=np.nan_to_num(self.compute_opacity(temperature,pressure,wngrid_filter))

        if wngrid is None:
            return orig
        else:
            # min_max =  (self.wavenumberGrid <= wngrid.max() ) & (self.wavenumberGrid >= wngrid.min())

            # total_bins = self.wavenumberGrid[min_max].shape[0]
            # if total_bins > wngrid.shape[0]:
            #     return np.append(np.histogram(self.wavenumberGrid,wngrid, weights=orig)[0]/np.histogram(self.wavenumberGrid,wngrid)[0],0)

            # else:
            return np.interp(wngrid, self.wavenumberGrid[wngrid_filter], orig)
