from .cia import CIA
import pickle
import numpy as np

class EndOfHitranCIAException(Exception):
    pass

class HitranCIA(CIA):

    def __init__(self,filename):
        super().__init__('HITRANCIA','None')

        self._filename = filename
        self._molecule_name = None
        self._wavenumber_grid = None
        self._temperature_grid = None
        self._xsec_grid = None
        self.load_hitran_file(filename)




    def load_hitran_file(self,filename):
        temp_list = []
        sigma_list = []

        with open(filename,'r') as f:
            #Read number of points
            while True:
                try:
                    start_wn,end_wn,total_points,T,max_cia=self.read_header(f)

                except EndOfHitranCIAException:
                    break
                #Append the temperature
                temp_list.append(T)
                #Clear the temporary list
                sigma_temp = []
                wn_temp = []
                for line in range(total_points):
                    _wn,_sigma = f.readline().split()
                    wn_temp.append(float(_wn))
                    sigma_temp.append(float(_sigma))
                
                #Ok we're done lets add the sigma
                sigma_list.append(np.array(sigma_temp))
                #set the wavenumber grid
                self._wavenumber_grid = np.array(wn_temp)
            self._temperature_grid = np.array(temp_list)
            self._xsec_grid = np.array(sigma_list)
            
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


    def read_header(self,f):
        line = f.readline()
        if line is None or line=='':
            raise EndOfHitranCIAException
        split = line.split()
        self._pair_name = split[0]
        start_wn = float(split[1])
        end_wn = float(split[2])
        total_points = int(split[3])
        T = float(split[4])
        max_cia = float(split[5])

        return start_wn,end_wn,total_points,T,max_cia
         


    @property
    def wavenumberGrid(self):
        return self._wavenumber_grid

    @property
    def temperatureGrid(self):
        return self._temperature_grid


    def compute_cia(self,temperature):
        return self.interp_linear_grid(temperature,*self.find_closest_temperature_index(temperature))