from .opacity import Opacity
import pickle
import numpy as np
import pathlib
class PickleOpacity(Opacity):
    """
    This is the base class for computing opactities

    """
    
    def __init__(self,filename):
        super().__init__('PickleOpacity:{}'.format(pathlib.Path(filename).stem[0:10]))

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self._resolution = None
        self._load_pickle_file(filename)

        
        
    @property
    def moleculeName(self):
        return self._molecule_name



    def _load_pickle_file(self,filename):

        #Load the pickle file
        self.info('Loading opacity from {}'.format(filename))
        with open(filename,'rb') as f:
            self._spec_dict = pickle.load(f)
        
        self._wavenumber_grid = self._spec_dict['wno']

        self._temperature_grid = self._spec_dict['t']
        self._pressure_grid = self._spec_dict['p']
        self._xsec_grid = self._spec_dict['xsecarr']
        self._resolution = np.average(np.diff(self._wavenumber_grid))
        self._molecule_name = self._spec_dict['name']
        self.clean_molecule_name()
    def clean_molecule_name(self):
        splits = self.moleculeName.split('_')
        self._molecule_name = splits[0].upper()

    @property
    def wavenumberGrid(self):
        return self._wavenumber_grid

    @property
    def temperatureGrid(self):
        return self._temperature_grid
    
    @property
    def pressureGrid(self):
        return self._pressure_grid

    @property
    def resolution(self):
        return self._resolution

    def find_closest_TP_index(self,temp,pressure):
        nearest_idx = np.abs(temp-self._temperature_grid).argmin() 
        t_idx_min = -1
        t_idx_max = -1
        if self._temperature_grid[nearest_idx] > temp:
            t_idx_max = nearest_idx
            t_idx_min = nearest_idx-1
        else:
            t_idx_min = nearest_idx
            t_idx_max = nearest_idx+1
            
        nearest_idx = np.abs(pressure-self._pressure_grid).argmin() 
        p_idx_min = -1
        p_idx_max = -1
        if self._pressure_grid[nearest_idx] > pressure:
            p_idx_max = nearest_idx
            p_idx_min = nearest_idx-1
        else:
            p_idx_min = nearest_idx
            p_idx_max = nearest_idx+1

        p_idx_min = max(0,p_idx_min)
        p_idx_max = min(len(self._pressure_grid)-1,p_idx_max)
        t_idx_min = max(0,t_idx_min)
        t_idx_max = min(len(self._temperature_grid)-1,t_idx_max)


        return t_idx_min,t_idx_max,p_idx_min,p_idx_max
    
    def interp_bilinear_grid(self,T,P,t_idx_min,t_idx_max,p_idx_min,p_idx_max):
        import numexpr as ne
        #FORMAT OF XSEC IS P,T,XSEC
        #P is x
        #T is y

        q_11 = self._xsec_grid[p_idx_min,t_idx_min]
        q_12 = self._xsec_grid[p_idx_min,t_idx_max]
        q_21 = self._xsec_grid[p_idx_max,t_idx_min]
        q_22 = self._xsec_grid[p_idx_max,t_idx_max]

        Tmax = self._temperature_grid[t_idx_max]
        Tmin = self._temperature_grid[t_idx_min]
        Pmax = self._pressure_grid[p_idx_max]
        Pmin = self._pressure_grid[p_idx_min]

        diff = ((Tmax-Tmin)*(Pmax-Pmin))
        if diff  == 0:
            return np.zeros_like(self._xsec_grid[0,0])
        factor = 1.0/((Tmax-Tmin)*(Pmax-Pmin))

        self.debug('FACTOR {}'.format(factor))

        return ne.evaluate('factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))')


        #return factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))



        








    def compute_opacity(self,temperature,pressure):
        return self.interp_bilinear_grid(temperature,pressure
                    ,*self.find_closest_TP_index(temperature,pressure)) / 10000