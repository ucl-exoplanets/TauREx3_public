from .opacity import Opacity
import pickle
import numpy as np
import pathlib
class PickleOpacity(Opacity):
    """
    This is the base class for computing opactities

    """
    
    def __init__(self,filename,interp_mode='linear'):
        super().__init__('PickleOpacity:{}'.format(pathlib.Path(filename).stem[0:10]))

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self._resolution = None
        self._load_pickle_file(filename)
        self._interp_mode = interp_mode
        
        
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
        self._pressure_grid = self._spec_dict['p']*1e5
        self._xsec_grid = self._spec_dict['xsecarr']
        self._resolution = np.average(np.diff(self._wavenumber_grid))
        self._molecule_name = self._spec_dict['name']

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()
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


    def find_closest_index(self,T,P):
        t_min=self._temperature_grid.searchsorted(T,side='right')-1
        t_max = t_min+1

        p_min=self._pressure_grid.searchsorted(P,side='right')-1
        p_max = p_min+1

        return t_min,t_max,p_min,p_max


  

    
    def interp_temp_only(self,T,t_idx_min,t_idx_max,P):
        Tmax = self._temperature_grid[t_idx_max]
        Tmin = self._temperature_grid[t_idx_min]
        fx0=self._xsec_grid[P,t_idx_min]
        fx1 = self._xsec_grid[P,t_idx_max]

        if self._interp_mode is 'linear':
            return fx0 + (fx1-fx0)*(T-Tmin)/(Tmax-Tmin)
        # else:
        #     alpha = 1/(1/Tmax - 1/Tmin)*(np.log(fx0)/np.log(fx1))
        #     beta = (T-Tmin)/(Tmax*T)

        #     return fx0*np.exp(alpha*beta)  


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
        #self.debug('Stats are {} {} {} {}'.format(self._temperature_grid[-1],self.pressureGrid[-1],self._max_temperature_id,self._max_pressure_id))
        if p_idx_max == 0 and t_idx_max == 0:

            return np.zeros_like(self._xsec_grid[0,0])

        check_pressure_max = P >= self._max_pressure
        check_temperature_max = T >= self._max_temperature

        check_pressure_min = P < self._min_pressure
        check_temperature_min = T < self._min_temperature


        self.debug('Check pressure min/max {}/{}'.format(check_pressure_min,check_pressure_max))
        self.debug('Check temeprature min/max {}/{}'.format(check_temperature_min,check_temperature_max))
        #Are we both max?
        if check_pressure_max and check_temperature_max:
            self.debug('Maximum Temperature pressure reached. Using last')
            return self._xsec_grid[-1,-1] 

        #Max pressure
        if check_pressure_max:
            self.debug('Max pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T,t_idx_min,t_idx_max,-1)
        
        #Max temperature
        if check_temperature_max:
            self.debug('Max temperature reached. Interpolating pressure only')
            return self.interp_pressure_only(P,p_idx_min,p_idx_max,-1)

        if check_pressure_min and check_temperature_min:
            return self._xsec_grid[0,0]
        
        if check_pressure_min:
            self.debug('Min pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T,t_idx_min,t_idx_max,0)          

        if check_temperature_min:
            self.debug('Min temeprature reached. Interpolating pressure only')
            return self.interp_pressure_only(P,p_idx_min,p_idx_max,0)  

        

        q_11 = self._xsec_grid[p_idx_min,t_idx_min]
        q_12 = self._xsec_grid[p_idx_min,t_idx_max]
        q_21 = self._xsec_grid[p_idx_max,t_idx_min]
        q_22 = self._xsec_grid[p_idx_max,t_idx_max]

        Tmax = self._temperature_grid[t_idx_max]
        Tmin = self._temperature_grid[t_idx_min]
        Pmax = self._pressure_grid[p_idx_max]
        Pmin = self._pressure_grid[p_idx_min]


        if self._interp_mode is 'linear':
            diff = ((Tmax-Tmin)*(Pmax-Pmin))
            factor = 1.0/((Tmax-Tmin)*(Pmax-Pmin))

            self.debug('FACTOR {}'.format(factor))

            return ne.evaluate('factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))')
        else:
            t_factor = 1.0/((1.0/Tmax) - (1.0/Tmin))
            b = ne.evaluate('t_factor*log(q_11/q_12)')
            a = ne.evaluate('q_11*exp(b/Tmin)')
            sigma = ne.evaluate('a*exp(-b/T)')

            b = ne.evaluate('t_factor*log(q_21/q_22)')
            a = ne.evaluate('q_21*exp(b/Tmin)')
            sigma_2 = ne.evaluate('a*exp(-b/T)')

            p_factor = (P-Pmin)/(Pmax-Pmin)
            return ne.evaluate('sigma + (sigma_2 - sigma)*p_factor')



        #return factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))



        








    def compute_opacity(self,temperature,pressure):

        return self.interp_bilinear_grid(temperature,pressure
                    ,*self.find_closest_index(temperature,pressure)) / 10000