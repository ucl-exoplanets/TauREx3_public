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

        self._max_pressure_id,self._max_temperature_id = len(self._pressure_grid)-1,len(self._temperature_grid)-1

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
    
    def interp_temp_only(self,T,t_idx_min,t_idx_max,P):
        Tmax = self._temperature_grid[t_idx_max]
        Tmin = self._temperature_grid[t_idx_min]
        fx0=self._xsec_grid[P,t_idx_min]
        fx1 = self._xsec_grid[P,t_idx_max]

        return fx0 + (fx1-fx0)*(T-Tmin)/(Tmax-Tmin)

    def interp_pressure_only(self,P,p_idx_min,p_idx_max,T):
        Pmax = self._pressure_grid[p_idx_max]
        Pmin = self._pressure_grid[p_idx_min]
        fx0=self._xsec_grid[p_idx_min,T]
        fx1 = self._xsec_grid[p_idx_max,T]

        return fx0 + (fx1-fx0)*(P-Pmin)/(Pmax-Pmin)


    def interp_bilinear_grid(self,T,P,t_idx_min,t_idx_max,p_idx_min,p_idx_max):
        import numexpr as ne
        #FORMAT OF XSEC IS P,T,XSEC
        #P is x
        #T is y

        self.debug('Interpolating {} {} {} {} {} {}'.format(T,P,t_idx_min,t_idx_max,p_idx_min,p_idx_max))
        self.debug('Stats are {} {} {} {}'.format(self._temperature_grid[-1],self.pressureGrid[-1],self._max_temperature_id,self._max_pressure_id))
        if p_idx_max == 0 and t_idx_max == 0:

            return np.zeros_like(self._xsec_grid[0,0])

        check_pressure_max = p_idx_min >= self._max_pressure_id
        check_temperature_max = t_idx_min >= self._max_temperature_id

        check_pressure_min = p_idx_max == 0
        check_temperature_min = t_idx_max == 0


        self.debug('Check pressure min/max {}/{}'.format(check_pressure_min,check_pressure_max))
        self.debug('Check temeprature min/max {}/{}'.format(check_temperature_min,check_temperature_max))
        #Are we both max?
        if check_pressure_max and check_temperature_max:
            self.warning('Maximum Temperature pressure reached. Using last')
            return self._xsec_grid[-1,-1] 

        #Max pressure
        if check_pressure_max:
            self.warning('Max pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T,t_idx_min,t_idx_max,-1)
        
        #Max temperature
        if check_temperature_max:
            self.warning('Max temperature reached. Interpolating pressure only')
            return self.interp_pressure_only(P,p_idx_min,p_idx_max,-1)

        if check_pressure_min and check_temperature_min:
            return self._xsec_grid[0,0]
        
        if check_pressure_min:
            self.warning('Min pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T,t_idx_min,t_idx_max,0)          

        if check_temperature_min:
            self.warning('Min temeprature reached. Interpolating pressure only')
            return self.interp_pressure_only(P,p_idx_min,p_idx_max,0)  

        

        q_11 = self._xsec_grid[p_idx_min,t_idx_min]
        q_12 = self._xsec_grid[p_idx_min,t_idx_max]
        q_21 = self._xsec_grid[p_idx_max,t_idx_min]
        q_22 = self._xsec_grid[p_idx_max,t_idx_max]

        Tmax = self._temperature_grid[t_idx_max]
        Tmin = self._temperature_grid[t_idx_min]
        Pmax = self._pressure_grid[p_idx_max]
        Pmin = self._pressure_grid[p_idx_min]

        diff = ((Tmax-Tmin)*(Pmax-Pmin))
        factor = 1.0/((Tmax-Tmin)*(Pmax-Pmin))

        self.debug('FACTOR {}'.format(factor))

        return ne.evaluate('factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))')


        #return factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))



        








    def compute_opacity(self,temperature,pressure):
        return self.interp_bilinear_grid(temperature,pressure
                    ,*self.find_closest_TP_index(temperature,pressure)) / 10000