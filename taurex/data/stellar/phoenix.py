from .star import BlackbodyStar
import numpy as np
import glob
import re
import os
import pathlib
import numpy as np
class PhoenixStar(BlackbodyStar):

    def __init__(self,temperature=5000,radius=1.0,phoenix_path=None):
        super().__init__(temperature=temperature,radius=radius)
        if phoenix_path is None:
            self.error('No file path to phoenix files defined')
            raise Exception('No file path to phoenix files defined')
        
        self.info('Star is PHOENIX type')
        self._phoenix_path = phoenix_path
        self.preload_phoenix_spectra()


    def preload_phoenix_spectra(self):
        
        T_list = self.detect_all_T(self._phoenix_path)



        self._temperature_grid = np.array([x[0] for x in T_list])
        self.debug('Detected temepratures = {}'.format(self._temperature_grid))
        self._avail_max_temp = max(self._temperature_grid)
        self._avail_min_temp = min(self._temperature_grid)
    
        self.debug('Temperature range = [{}-{}] '.format(self._avail_min_temp,self._avail_max_temp))
        self._max_index = self._temperature_grid.shape[0]
        self.spectra_grid = []

        #Load in all arrays
        for temp,f in T_list:
            self.debug('Loading {} {}'.format(temp,f))
            arr = np.loadtxt(f)
            self.wngrid = np.copy(arr[:,0])
            self.spectra_grid.append(arr[:,1]*10.0) #To SI



    
    def find_closest_index(self,T):
        t_min=self._temperature_grid.searchsorted(T,side='right')
        t_max = t_min+1

        return t_min,t_max

    
    def interpolate_linear_temp(self,T):
        t_min,t_max = self.find_closest_index(T)
        if self._temperature_grid[t_min] == T:
            return self.spectra_grid[t_min]

        Tmax = self._temperature_grid[t_max]
        Tmin = self._temperature_grid[t_min]
        fx0=self.spectra_grid[t_min]
        fx1 = self.spectra_grid[t_max]

        return fx0 + (fx1-fx0)*(T-Tmin)/(Tmax-Tmin)



    def detect_all_T(self,path):
        files = glob.glob(os.path.join(self._phoenix_path,'*.fmt'))
        files.sort()

        temp_list = []
        for f in files:
            #Gewt just the name of the file
            clean_name = pathlib.Path(f).stem
            #Split it by numbers
            split = re.split('(\d+)',clean_name)
            try:
                _T = float(split[1])
            except Exception:
                self.warning('Problem when reading filename {}'.format(f))
                continue
            temp_list.append( (_T,f) )
        
        #Now sort the numbers
        temp_list.sort(key=lambda x: x[0])
        return temp_list

    
    def initialize(self,wngrid):
        #If temperature outside of range, use blavkbody
        if self.temperature > self._avail_max_temp or self.temperature < self._avail_min_temp:
            self.warning('Using black body as temperature is outside of Star temeprature range {}'.format(self.temperature))
            super().initialize(wngrid)
        else:
            sed = self.interpolate_linear_temp(self.temperature)
            self.sed = np.interp(wngrid,self.wngrid,sed)
            