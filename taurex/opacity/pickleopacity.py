from .opacity import Opacity
import pickle
class PickleOpacity(Opacity):
    """
    This is the base class for computing opactities

    """
    
    def __init__(self,filename):
        super().__init__('PickleOpacity')

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
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
        self._pressure_grid = self._spec_dict['t']
        self._xsec_grid = self._spec_dict['xsecarr']
        self._molecule_name = self._spec_dict['name']

    @property
    def wavenumberGrid(self):
        return self._wavenumber_grid

    def find_closest_TP_index(self,temp,pressure):
        t_idxmax = self._temperature_grid[self._temperature_grid > temp].argmin()
        t_idxmin = self._temperature_grid[self._temperature_grid <= temp].argmax()
        
        p_idxmax = self._pressure_grid[self._pressure_grid > pressure].argmin()
        p_idxmin = self._pressure_grid[self._pressure_grid <= pressure].argmax()

        return t_idxmin,t_idxmax,p_idxmin,p_idxmax
    
    def interp_bilinear_grid(self,T,P,t_idx_min,t_idx_max,p_idx_min,p_idx_max):

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
        factor = 1.0/((Tmax-Tmin)*(Pmax-Pmin))
    
        return factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))



        








    def opacity(self,temperature,pressure):
        return self.interp_bilinear_grid(temperature,pressure
                    ,*self.find_closest_TP_index(temperature,pressure))