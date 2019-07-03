
from .star import BlackbodyStar
import numpy as np
import glob
import re
import os
import pathlib
import numpy as np
from taurex.util.math import interp_lin_only
class PhoenixStar(BlackbodyStar):
    """
    A star that uses the `PHOENIX <https://www.aanda.org/articles/aa/abs/2013/05/aa19058-12/aa19058-12.html>`_
    synthetic stellar atmosphere spectrums.

    These spectrums are read from ``.fmt`` files in a directory given by ``phoenix_path``
    Each file must contain the spectrum for one temperature

    Parameters
    ----------

    temperature : float
        Stellar temperature in Kelvin
    
    radius : float
        Stellar radius in Solar radius
    
    phoenix_path : str
        Path to phoenix ``.fmt`` files

    
    Raises
    ------
    Exception
        Raised when no phoenix path is defined


    """





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
        self.debug('Detected temepratures = %s',self._temperature_grid)
        self._avail_max_temp = max(self._temperature_grid)
        self._avail_min_temp = min(self._temperature_grid)
    
        self.debug('Temperature range = [%s-%s] ',self._avail_min_temp,self._avail_max_temp)
        self._max_index = self._temperature_grid.shape[0]
        self.spectra_grid = []

        #Load in all arrays
        for temp,f in T_list:
            self.debug('Loading %s %s',temp,f)
            arr = np.loadtxt(f)
            self.wngrid = 10000/np.copy(arr[:,0])
            self.spectra_grid.append(arr[:,1]*10.0) #To SI



    
    def find_closest_index(self,T):
        """
        Finds the two closest indices in our temeprature grid to our desired temperature

        Parameters
        ----------
        T : float
            Temperature in Kelvin

        Returns
        -------
        t_min : int
            Index to the left of ``T``
        
        t_max : int
            Index to the right of ``T``

        """


        t_min=self._temperature_grid.searchsorted(T,side='right')-1
        t_max = t_min+1

        return t_min,t_max

    
    def interpolate_linear_temp(self,T):
        """
        Linearly interpolates the spectral emission density grid to the 
        temperature given by ``T``

        Parameters
        ----------
        T : float 
            Temeprature to interpolate to
        
        Returns
        -------
        out : :obj:`array`
            Spectral emission density interpolated to desired temperature
        

        """
        t_min,t_max = self.find_closest_index(T)
        if self._temperature_grid[t_min] == T:
            return self.spectra_grid[t_min]

        Tmax = self._temperature_grid[t_max]
        Tmin = self._temperature_grid[t_min]
        fx0=self.spectra_grid[t_min]
        fx1 = self.spectra_grid[t_max]

        return interp_lin_only(fx0,fx1,T,Tmin,Tmax)



    def detect_all_T(self,path):
        """
        Finds files and detects all temperatures available in path

        Parameters
        ----------
        path : str
            Path to directory containing PHOENIX data

        """
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
                self.warning('Problem when reading filename %s',f)
                continue
            temp_list.append( (_T,f) )
        
        #Now sort the numbers
        temp_list.sort(key=lambda x: x[0])
        return temp_list

    
    def initialize(self,wngrid):
        """
        Initializes and interpolates the spectral emission density to the current
        stellar temperature and given wavenumber grid

        Parameters
        ----------
        wngrid : :obj:`array`
            Wavenumber grid to interpolate the SED to
        
        """
        #If temperature outside of range, use blavkbody
        if self.temperature > self._avail_max_temp or self.temperature < self._avail_min_temp:
            self.warning('Using black body as temperature is outside of Star temeprature range {}'.format(self.temperature))
            super().initialize(wngrid)
        else:
            sed = self.interpolate_linear_temp(self.temperature)
            self.sed = np.interp(wngrid,self.wngrid,sed)
            