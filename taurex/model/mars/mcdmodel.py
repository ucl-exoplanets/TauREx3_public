from taurex.log import Logger
import numpy as np
import math
from ..model import ForwardModel
from taurex.util import bindown
from taurex.data.stellar import BlackbodyStar
from datetime import datetime
from taurex.data.planet import Mars

from taurex.data.profiles.chemistry.chemistry import Chemistry

class MCDChemistry(Chemistry):
    
    gases = ['CO2','N2','Ar','CO','O','O2','O3','H','H2','He']
    
    def __init__(self,mix_array):
        from taurex.cache import OpacityCache
        Chemistry.__init__(self,'MCD')
        self.molecules_i_have = OpacityCache().find_list_of_molecules()

        
        self.arrange_and_build(mix_array)
    def arrange_and_build(self,mixarray):
        
        active_gases = [(idx,x) for idx,x in enumerate(self.gases) if x in self.molecules_i_have]
        inactive_gases = [(idx,x) for idx,x in enumerate(self.gases) if not x in self.molecules_i_have]
        
        self._active = [x for idx,x in active_gases]
        self._inactive = [x for idx,x in inactive_gases]
    
        num_active = len(active_gases)
        num_inactive = len(inactive_gases)
        
        num_layers = mixarray.shape[1]
        
        self.active_mixratio_profile = np.zeros(shape=(num_active,num_layers))
        self.inactive_mixratio_profile   = np.zeros(shape=(num_inactive,num_layers))
        
        for idx,val in enumerate(active_gases):
            mix_idx,gas = val
            self.active_mixratio_profile[idx,:] = mixarray[mix_idx,:]

        for idx,val in enumerate(inactive_gases):
            mix_idx,gas = val
            self.inactive_mixratio_profile[idx,:] = mixarray[mix_idx,:]            
    @property
    def activeGases(self):
        return self._active


    @property
    def inactiveGases(self):
        return  self._inactive
            
            
    @property
    def activeGasMixProfile(self):
        """
        Active gas layer by layer mix profile

        Returns
        -------
        active_mix_profile : :obj:`array`

        """
        return self.active_mixratio_profile

    @property
    def inactiveGasMixProfile(self):
        """
        Inactive gas layer by layer mix profile

        Returns
        -------
        inactive_mix_profile : :obj:`array`

        """
        return self.inactive_mixratio_profile    

class MarsMCDModel(ForwardModel):

    def __init__(self,data_dir,
                 latitude=40,
                 longitude=40,
                 date_type=0,
                 date=datetime.now(),
                 localtime=0.0,
                 hires=False,
                 minimum_pressure=1e-5,
                 nlayers=100):
        ForwardModel.__init__(self,'MCDModel')
        self._data_dir = data_dir
        self.latitude = latitude
        self.longitude = longitude
        self.date_type = date_type
        self.date= date
        self.localtime = localtime
        self.hires = hires
        self.min_pressure = minimum_pressure
        self.nlayers = nlayers
        
        self._planet = Mars()
        self._star = BlackbodyStar()
    def extra_array(self):
        extra = np.zeros(100,dtype='i')
        #surface pressure
        extra[18] = 1
        #Mixing ratios
        extra[56:65] = 1
        extra[77] = 1
        
        return extra
        
        
    def build(self):
        from pymcd import call_mcd,VertCoordType,DateType,Scenario,PerturbationType

        res = call_mcd(self._data_dir,
                    vert_coord_type=VertCoordType.Pressure,
                    vert_coord=100.0,
                    longitude=self.longitude,
                    latitude=self.latitude,
                    date_type=self.date_type,
                    date=self.date,
                    localtime=self.localtime,
                    enable_hires=self.hires,
                    scenario  = Scenario.ClimateAvgSolar,
                    perturb_type = PerturbationType.NONE,
                    seed_or_sigma_mul = 0.0,
                    gwlength = 0.0,
                    include_extra = self.extra_array())
        
        self._surface_pressure = res[-1][18]

        
        self.build_profiles()
        self.compute_altitude_gravity_scaleheight_profile()
    
    def build_profiles(self):
        from taurex.data.profiles.pressure import SimplePressureProfile
        from pymcd import call_mcd,VertCoordType,DateType,Scenario,PerturbationType
        
        self.pressure = SimplePressureProfile(nlayers=self.nlayers,atm_min_pressure=1e-4,
                                                          atm_max_pressure=self._surface_pressure)
        self.pressure.compute_pressure_profile()
        temperature=[]
        mix_ratios = []
        density = []
        particles = []
        extra_array = self.extra_array()
        for pressure in self.pressureProfile:
            gas_mix = np.zeros(10)
            res = call_mcd(self._data_dir,
                        vert_coord_type=VertCoordType.Pressure,
                        vert_coord=pressure,
                        longitude=self.longitude,
                        latitude=self.latitude,
                        date_type=self.date_type,
                        date=self.date,
                        localtime=self.localtime,
                        enable_hires=self.hires,
                        scenario  = Scenario.ClimateAvgSolar,
                        perturb_type = PerturbationType.NONE,
                        seed_or_sigma_mul = 0.0,
                        gwlength = 0.0,
                        include_extra = extra_array)
            
            temperature.append(res[2])
            density.append(res[1])
            extra_part = res[-1]
            gas_mix[0:9] = extra_part[56:65]
            gas_mix[9] = extra_part[77]
            
            mix_ratios.append(gas_mix)
            
        mix_ratios = np.vstack(mix_ratios).T
        
        
        self._temperature_profile = np.array(temperature)
        
        self._chemistry = MCDChemistry(mix_ratios)
        self._chemistry.initialize_chemistry(self.nlayers,0,0,0)
    @property
    def pressureProfile(self):
        return self.pressure.profile

    @property
    def temperatureProfile(self):
        return self._temperature_profile

    @property
    def densityProfile(self):
        from taurex.constants import KBOLTZ
        return (self.pressureProfile)/(KBOLTZ*self.temperatureProfile)


    @property
    def altitudeProfile(self):
        return self.altitude_profile
    
    @property
    def chemistry(self):
        return self._chemistry
        
        
    # altitude, gravity and scale height profile
    def compute_altitude_gravity_scaleheight_profile(self):
        from taurex.constants import KBOLTZ
        
        
        mu_profile = self._chemistry.muProfile
        # build the altitude profile from the bottom up
        nlayers = self.pressure.nLayers
        H = np.zeros(nlayers)
        g = np.zeros(nlayers)
        z = np.zeros(nlayers)


        g[0] = self._planet.gravity # surface gravity (0th layer)
        H[0] = (KBOLTZ*self.temperatureProfile[0])/(mu_profile[0]*g[0]) # scaleheight at the surface (0th layer)

        for i in range(1, nlayers):
            deltaz = (-1.)*H[i-1]*np.log(self.pressure.pressure_profile_levels[i]/self.pressure.pressure_profile_levels[i-1])
            z[i] = z[i-1] + deltaz # altitude at the i-th layer

            with np.errstate(over='ignore'):
                g[i] = self._planet.gravity_at_height(z[i]) # gravity at the i-th layer
                #print('G[{}] = {}'.format(i,g[i]))
            with np.errstate(divide='ignore'):
                H[i] = (KBOLTZ*self.temperatureProfile[i])/(mu_profile[i]*g[i])

        self.altitude_profile = z
        self.scaleheight_profile = H
        self.gravity_profile = g        
        
    @property
    def star(self):
        return self._star
    
    @property
    def planet(self):
        return self._planet