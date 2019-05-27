from taurex.log import Logger
import numpy as np
from .opacity import Opacity
import numexpr as ne
class InterpolatingOpacity(Opacity):
    """
    Provides interpolation methods

    """
    
    def __init__(self,name,interpolation_mode='linear'):
        super().__init__(name)
        self._interp_mode = interpolation_mode

    @property
    def xsecGrid(self):
        raise NotImplementedError

    

    def find_closest_index(self,T,P):
        t_min=self.temperatureGrid.searchsorted(T,side='right')-1
        t_max = t_min+1

        p_min=self.pressureGrid.searchsorted(P,side='right')-1
        p_max = p_min+1

        return t_min,t_max,p_min,p_max


    def set_interpolation_mode(self,interp_mode):
        self._interp_mode = interp_mode

    
    def interp_temp_only(self,T,t_idx_min,t_idx_max,P):
        Tmax = self.temperatureGrid[t_idx_max]
        Tmin = self.temperatureGrid[t_idx_min]
        fx0=self.xsecGrid[P,t_idx_min]
        fx1 = self.xsecGrid[P,t_idx_max]

        if self._interp_mode is 'linear':
            return fx0 + (fx1-fx0)*(T-Tmin)/(Tmax-Tmin)
        else:
            alpha = ne.evaluate('(1/(1/Tmax - 1/Tmin)) *(log(fx0/fx1))')
            beta = (T-Tmin)/(Tmax*T)

            return ne.evaluate('fx0*exp(alpha*beta)')  


    def interp_pressure_only(self,P,p_idx_min,p_idx_max,T):
        Pmax = self.pressureGrid[p_idx_max]
        Pmin = self.pressureGrid[p_idx_min]
        fx0=self.xsecGrid[p_idx_min,T]
        fx1 = self.xsecGrid[p_idx_max,T]

        return fx0 + (fx1-fx0)*(P-Pmin)/(Pmax-Pmin)


    def interp_bilinear_grid(self,T,P,t_idx_min,t_idx_max,p_idx_min,p_idx_max):
        import numexpr as ne
        

        self.debug('Interpolating {} {} {} {} {} {}'.format(T,P,t_idx_min,t_idx_max,p_idx_min,p_idx_max))
        #self.debug('Stats are {} {} {} {}'.format(self.temperatureGrid[-1],self.pressureGrid[-1],self._max_temperature_id,self._max_pressure_id))
        if p_idx_max == 0 and t_idx_max == 0:

            return np.zeros_like(self.xsecGrid[0,0])

        check_pressure_max = P >= self._max_pressure
        check_temperature_max = T >= self._max_temperature

        check_pressure_min = P < self._min_pressure
        check_temperature_min = T < self._min_temperature


        self.debug('Check pressure min/max {}/{}'.format(check_pressure_min,check_pressure_max))
        self.debug('Check temeprature min/max {}/{}'.format(check_temperature_min,check_temperature_max))
        #Are we both max?
        if check_pressure_max and check_temperature_max:
            self.debug('Maximum Temperature pressure reached. Using last')
            return self.xsecGrid[-1,-1] 

        #Max pressure
        if check_pressure_max:
            self.debug('Max pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T,t_idx_min,t_idx_max,-1)
        
        #Max temperature
        if check_temperature_max:
            self.debug('Max temperature reached. Interpolating pressure only')
            return self.interp_pressure_only(P,p_idx_min,p_idx_max,-1)

        if check_pressure_min and check_temperature_min:
            return self.xsecGrid[0,0]
        
        if check_pressure_min:
            self.debug('Min pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T,t_idx_min,t_idx_max,0)          

        if check_temperature_min:
            self.debug('Min temeprature reached. Interpolating pressure only')
            return self.interp_pressure_only(P,p_idx_min,p_idx_max,0)  

        

        q_11 = self.xsecGrid[p_idx_min,t_idx_min]
        q_12 = self.xsecGrid[p_idx_min,t_idx_max]
        q_21 = self.xsecGrid[p_idx_max,t_idx_min]
        q_22 = self.xsecGrid[p_idx_max,t_idx_max]

        Tmax = self.temperatureGrid[t_idx_max]
        Tmin = self.temperatureGrid[t_idx_min]
        Pmax = self.pressureGrid[p_idx_max]
        Pmin = self.pressureGrid[p_idx_min]


        if self._interp_mode is 'linear':
            diff = ((Tmax-Tmin)*(Pmax-Pmin))
            factor = 1.0/((Tmax-Tmin)*(Pmax-Pmin))

            self.debug('FACTOR {}'.format(factor))

            return ne.evaluate('factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))')
        else:
            t_factor =(1/(1/Tmax - 1/Tmin)) 
            alpha = ne.evaluate('t_factor*log(q_11/q_12)')
            beta = (T-Tmin)/(Tmax*T)

            sigma=ne.evaluate('q_11*exp(alpha*beta)')  

            alpha = ne.evaluate('t_factor*log(q_21/q_22)')
            sigma_2 = ne.evaluate('q_21*exp(alpha*beta)')

            p_factor = (P-Pmin)/(Pmax-Pmin)
            return ne.evaluate('sigma + (sigma_2 - sigma)*p_factor')   
    

    def compute_opacity(self,temperature,pressure):

        return self.interp_bilinear_grid(temperature,pressure
                    ,*self.find_closest_index(temperature,pressure)) / 10000