from .cia import CIA
import pickle
import numpy as np
from pathlib import Path
class PickleCIA(CIA):
    """
    This is the base class for computing opactities

    """
    
    def __init__(self,filename,pair_name=None):

        if pair_name is None:
            pair_name=Path(filename).stem
            
        super().__init__('PickleCIA',pair_name)

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self._load_pickle_file(filename)

    def _load_pickle_file(self,filename):

        #Load the pickle file
        self.info('Loading cia cross section from %s',filename)
        with open(filename,'rb') as f:
            self._spec_dict = pickle.load(f, encoding='latin1')
        
        self._wavenumber_grid = self._spec_dict['wno']
        self._temperature_grid = self._spec_dict['t']
        self._xsec_grid = self._spec_dict['xsecarr']


    @property
    def wavenumberGrid(self):
        return self._wavenumber_grid

    @property
    def temperatureGrid(self):
        return self._temperature_grid


    def find_closest_temperature_index(self,temperature):
        nearest_idx = np.abs(temperature-self.temperatureGrid).argmin() 
        t_idx_min = -1
        t_idx_max = -1
        if self._temperature_grid[nearest_idx] > temperature:
            t_idx_max = nearest_idx
            t_idx_min = nearest_idx-1
        else:
            t_idx_min = nearest_idx
            t_idx_max = nearest_idx+1
        return t_idx_min,t_idx_max
    

    def interp_linear_grid(self,T,t_idx_min,t_idx_max):
        Tmax = self._temperature_grid[t_idx_max]
        Tmin = self._temperature_grid[t_idx_min]
        fx0=self._xsec_grid[t_idx_min]
        fx1 = self._xsec_grid[t_idx_max]

        return fx0 + (fx1-fx0)*(T-Tmin)/(Tmax-Tmin)


    def compute_cia(self,temperature):
        return self.interp_linear_grid(temperature,*self.find_closest_temperature_index(temperature))

    
