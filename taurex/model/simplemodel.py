from taurex.log import Logger
import numpy as np
import math
from .model import ForwardModel
from taurex.util import bindown

class SimpleForwardModel(ForwardModel):
    """ A 'simple' base model in the sense that its just
    a fairly standard single profiles model. Most like you'll
    inherit from this to do your own fuckery
    
    Parameters
    ----------
    name: string
        Name to use in logging
    
    planet: :obj:`Planet` or :obj:`None`
        Planet object created or None to use the default planet (Jupiter)

    
    """
    def __init__(self,name,
                            planet=None,
                            star=None,
                            pressure_profile=None,
                            temperature_profile=None,
                            chemistry=None,
                            nlayers=100,
                            atm_min_pressure=1e-4,
                            atm_max_pressure=1e6,

                            ):
        super().__init__(name)
        


        self._planet = planet
        self._star=star
        self._pressure_profile = pressure_profile
        self._temperature_profile = temperature_profile
        self._chemistry = chemistry
        self.debug('Passed: %s %s %s %s %s',planet,star,pressure_profile,temperature_profile,chemistry)
        self.altitude_profile=None
        self.scaleheight_profile=None
        self.gravity_profile=None
        self.setup_defaults(nlayers,atm_min_pressure,atm_max_pressure)

        self._initialized = False

        self._sigma_opacities = None

        self._native_grid = None


    def _compute_inital_mu(self):
        from taurex.data.profiles.chemistry import TaurexChemistry,ConstantGas
        tc = TaurexChemistry()
        tc.addGas(ConstantGas('H2O'))
        self._inital_mu=tc

    




        



    def setup_defaults(self,nlayers,atm_min_pressure,atm_max_pressure):

        if self._pressure_profile is None:
            from taurex.data.profiles.pressure import SimplePressureProfile
            self.info('No pressure profile defined, using simple pressure profile with')
            self.info('parameters nlayers: %s, atm_pressure_range=(%s,%s)',nlayers,atm_min_pressure,atm_max_pressure)
            self._pressure_profile = SimplePressureProfile(nlayers,atm_min_pressure,atm_max_pressure)

        if self._planet is None:
            from taurex.data import Planet
            self.warning('No planet defined, using Jupiter as planet')
            self._planet = Planet()

        if self._temperature_profile is None:
            from taurex.data.profiles.temperature import Isothermal
            self.warning('No temeprature profile defined using default Isothermal profile with T=1500 K')
            self._temperature_profile = Isothermal()


        if self._chemistry is None:
            from taurex.data.profiles.chemistry import TaurexChemistry,ConstantGas
            tc = TaurexChemistry()
            self.warning('No gas profile set, using constant profile with H2O and CH4')
            tc.addGas(ConstantGas('H2O',mix_ratio=1e-5))
            tc.addGas(ConstantGas('CH4',mix_ratio=1e-6))
            self._chemistry = tc

        if self._star is None:
            from taurex.data.stellar import BlackbodyStar
            self.warning('No star, using the Sun')
            self._star = BlackbodyStar()
    def initialize_profiles(self):
        self.info('Computing pressure profile')
        
        self.pressure.compute_pressure_profile()
        
        self._temperature_profile.initialize_profile(self._planet,
                    self.pressure.nLayers,
                    self.pressure.profile)
        
        #Initialize the atmosphere with a constant gas profile
        if self._initialized is False:
            self._inital_mu.initialize_chemistry(self.pressure.nLayers,
                                                self.temperatureProfile,self.pressureProfile,
                                                None)
            self.compute_altitude_gravity_scaleheight_profile(self._inital_mu.muProfile)
            self._initialized=True
        
        #Now initialize the gas profile
        self._chemistry.initialize_chemistry(self.pressure.nLayers,
                                                self.temperatureProfile,self.pressureProfile,
                                                self.altitude_profile)
        
        #Compute gravity scale height
        self.compute_altitude_gravity_scaleheight_profile()

        #quit()
    




    def collect_fitting_parameters(self):
        self._fitting_parameters = {}
        self._fitting_parameters.update(self.fitting_parameters())
        self._fitting_parameters.update(self._planet.fitting_parameters())
        if self._star is not None:
            self._fitting_parameters.update(self._star.fitting_parameters())
        self._fitting_parameters.update(self.pressure.fitting_parameters())
        self._fitting_parameters.update(self._temperature_profile.fitting_parameters())
        self._fitting_parameters.update(self._chemistry.fitting_parameters())

        for contrib in self.contribution_list:
            self._fitting_parameters.update(contrib.fitting_parameters())

        self.debug('Available Fitting params: %s',list(self._fitting_parameters.keys()))
    def build(self):
        self.info('Building model........')
        self._compute_inital_mu()
        self.info('Collecting paramters')
        self.collect_fitting_parameters()
        self.info('Setting up profiles')
        self.initialize_profiles()
        

        self.info('Setting up contributions')
        for contrib in self.contribution_list:
            contrib.build(self)
        self.info('DONE')




    # altitude, gravity and scale height profile
    def compute_altitude_gravity_scaleheight_profile(self,mu_profile=None):
        from taurex.constants import KBOLTZ
        if mu_profile is None:
            mu_profile=self._chemistry.muProfile

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
    def pressureProfile(self):
        return self.pressure.profile

    @property
    def temperatureProfile(self):
        return self._temperature_profile.profile

    @property
    def densityProfile(self):
        from taurex.constants import KBOLTZ
        return (self.pressureProfile)/(KBOLTZ*self.temperatureProfile)


    @property
    def altitudeProfile(self):
        return self.altitude_profile

    @property
    def nLayers(self):
        return self.pressure.nLayers

    @property
    def chemistry(self):
        return self._chemistry
    
    @property
    def pressure(self):
        return self._pressure_profile

    @property
    def temperature(self):
        return self._temperature_profile

    @property
    def star(self):
        return self._star
    
    @property
    def planet(self):
        return self._planet


    @property
    def nativeWavenumberGrid(self):
        from taurex.cache.opacitycache import OpacityCache
        wavenumbergrid = [OpacityCache()[gas].wavenumberGrid for gas in self.chemistry.activeGases]

        current_grid = None
        for wn in wavenumbergrid:
            if current_grid is None:
                current_grid = wn
            if wn.shape[0] > current_grid.shape[0]:
                current_grid = wn
        
        return current_grid
                

                


    def model(self,wngrid=None,return_contrib=False):
        self.initialize_profiles()

        native_grid = self.nativeWavenumberGrid
        if wngrid is not None:
            wn_min = wngrid.min()*0.9
            wn_max = wngrid.max()*1.1
            native_filter = (native_grid >= wn_min) & (native_grid <= wn_max)
            native_grid = native_grid[native_filter]


        self._star.initialize(native_grid)
        for contrib in self.contribution_list:
            contrib.prepare(self,native_grid)
        absorp,tau,contrib = self.path_integral(native_grid,return_contrib)
        if wngrid is None:
            return absorp,absorp,tau,contrib
        else:
            new_absp = bindown(native_grid,absorp,wngrid)
            return new_absp,absorp,tau,contrib

    def path_integral(self,wngrid,return_contrib):
        raise NotImplementedError

    def write(self,output):

        #Run a model if needed
        self.model(self.nativeWavenumberGrid)

        model = super().write(output)

        #Write Gas

        self._chemistry.write(model)
        self._temperature_profile.write(model)
        self._planet.write(model)
        self.pressure.write(model)
        self._star.write(model)
        

    
        return model