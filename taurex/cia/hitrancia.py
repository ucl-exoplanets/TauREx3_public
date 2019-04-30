from .cia import CIA
import pickle
import numpy as np

class EndOfHitranCIAException(Exception):
    pass


def hashwn(start_wn,end_wn):
    return str(start_wn)+str(end_wn)

class HitranCiaGrid(object):

    def __init__(self,wn_min,wn_max):
        self.wn = None
        self.Tsigma=[]

    def add_temperature(self,T,sigma):
        self.Tsigma.append((T,sigma))
    
    @property
    def temperature(self):
        return [t for t,s in self.Tsigma]

    @property
    def sigma(self):
        return  [s for t,s in self.Tsigma]

    def find_closest_temperature_index(self,temperature):
        temp_grid = np.array(self.temperature)
        nearest_idx = np.abs(temperature-temp_grid ).argmin()
        t_idx_min = -1
        t_idx_max = -1
        if temp_grid[nearest_idx] > temperature:
            t_idx_max = nearest_idx
            t_idx_min = nearest_idx-1
        else:
            t_idx_min = nearest_idx
            t_idx_max = nearest_idx+1
        return t_idx_min,t_idx_max
    

    def interp_linear_grid(self,T,t_idx_min,t_idx_max):
        temp_grid = np.array(self.temperature)
        Tmax = temp_grid[t_idx_max]
        Tmin = temp_grid[t_idx_min]
        fx0=self.sigma[t_idx_min]
        fx1 = self.sigma[t_idx_max]

        return fx0 + (fx1-fx0)*(T-Tmin)/(Tmax-Tmin)  

    def sortTempSigma(self):
        import operator
        self.Tsigma.sort(key = operator.itemgetter(0))

    def fill_temperature(self,temperatures):
        
        for t in temperatures:
            if t in self.temperature:
                continue
            #print(t,self.temperature)
            if t < min(self.temperature) or t > max(self.temperature):
                self.add_temperature(t,np.zeros_like(self.wn))
            else:
                self.add_temperature(t,self.interp_linear_grid(t,*self.find_closest_temperature_index(t)))
            self.sortTempSigma()


class HitranCIA(CIA):

    def __init__(self,filename):
        super().__init__('HITRANCIA','None')

        self._filename = filename
        self._molecule_name = None
        self._wavenumber_grid = None
        self._temperature_grid = None
        self._xsec_grid = None
        self._wn_dict = {}
        self.load_hitran_file(filename)




    def load_hitran_file(self,filename):
        temp_list = []


        with open(filename,'r') as f:
            #Read number of points
            while True:
                try:
                    start_wn,end_wn,total_points,T,max_cia=self.read_header(f)

                except EndOfHitranCIAException:
                    break
                if not T in temp_list:
                    temp_list.append(T)
                
                wn_hash=hashwn(start_wn,end_wn)

                wn_obj = None
                if not wn_hash in self._wn_dict:
                    self._wn_dict[wn_hash] = HitranCiaGrid(start_wn,end_wn)
                
                wn_obj=self._wn_dict[wn_hash]

                #Clear the temporary list
                sigma_temp = []
                wn_temp = []
                for x in range(total_points):
                    line = f.readline()
                    #print(line)
                    splits = line.split()

                    _wn = splits[0]
                    _sigma=splits[1]
                    wn_temp.append(float(_wn))
                    sigma_temp.append(float(_sigma))
                
                #Ok we're done lets add the sigma
                wn_obj.add_temperature(T,np.array(sigma_temp))
                #set the wavenumber grid
                wn_obj.wn = np.array(wn_temp)
            #self._temperature_grid = np.array(temp_list)
            #self._xsec_grid = np.array(sigma_list)
        temp_list.sort()
        self._temperature_grid = np.array(temp_list)
        self.fill_gaps(temp_list)
        self.compute_final_grid()
    def fill_gaps(self,temperature):
        
        for wn_obj in self._wn_dict.values():
            wn_obj.sortTempSigma()
            wn_obj.fill_temperature(temperature)


    def compute_final_grid(self):
        _wngrid = []
        for w in self._wn_dict.values():
            _wngrid.append(w.wn)
        self._wavenumber_grid = np.concatenate(_wngrid)
        sorted_idx = np.argsort(self._wavenumber_grid )
        self._wavenumber_grid = self._wavenumber_grid[sorted_idx]
        _sigma_array = []



        for idx,t in enumerate(self._temperature_grid):
            _temp_sigma=[]
            for w in self._wn_dict.values():
                _temp_sigma.append(w.Tsigma[idx][1])
            


            _sigma_array.append(np.concatenate(_temp_sigma)[sorted_idx])

        self._xsec_grid = np.array(_sigma_array)



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