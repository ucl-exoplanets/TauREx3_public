from taurex.log import Logger
import numpy as np
import math
from .model import ForwardModel


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
                            gas_profile=None,
                            nlayers=100,
                            atm_min_pressure=1e-4,
                            atm_max_pressure=1e6,

                            ):
        super().__init__(name)
        


        self._planet = planet
        self._star=star
        self.pressure_profile = pressure_profile
        self._temperature_profile = temperature_profile
        self._gas_profile = gas_profile
        self.debug('Passed: {} {} {} {} {}'.format(planet,star,pressure_profile,temperature_profile,gas_profile))
        self.altitude_profile=None
        self.scaleheight_profile=None
        self.gravity_profile=None
        self.setup_defaults(nlayers,atm_min_pressure,atm_max_pressure)

        self._initialized = False

        self._sigma_opacities = None

        self._native_grid = None


    def _compute_inital_mu(self):
        from taurex.data.profiles.gas import ConstantGasProfile
        self._inital_mu=ConstantGasProfile()

    




        



    def setup_defaults(self,nlayers,atm_min_pressure,atm_max_pressure):

        if self.pressure_profile is None:
            from taurex.data.profiles.pressure import SimplePressureProfile
            self.info('No pressure profile defined, using simple pressure profile with')
            self.info('parameters nlayers: {}, atm_pressure_range=({},{})'.format(nlayers,atm_min_pressure,atm_max_pressure))
            self.pressure_profile = SimplePressureProfile(nlayers,atm_min_pressure,atm_max_pressure)

        if self._planet is None:
            from taurex.data import Planet
            self.warning('No planet defined, using Jupiter as planet')
            self._planet = Planet()

        if self._temperature_profile is None:
            from taurex.data.profiles.temperature import Isothermal
            self.warning('No temeprature profile defined using default Isothermal profile with T=1500 K')
            self._temperature_profile = Isothermal()


        if self._gas_profile is None:
            from taurex.data.profiles.gas import ConstantGasProfile
            self.warning('No gas profile set, using constant profile with H2O and CH4')
            self._gas_profile = ConstantGasProfile()

        if self._star is None:
            from taurex.data.stellar import Star
            self.warning('No star, using the Sun')
            self._star = Star()
    def initialize_profiles(self):
        self.info('Computing pressure profile')
        
        self.pressure_profile.compute_pressure_profile()
        
        self._temperature_profile.initialize_profile(self._planet,
                    self.pressure_profile.nLayers,
                    self.pressure_profile.profile)
        
        #Initialize the atmosphere with a constant gas profile
        if self._initialized is False:
            self._inital_mu.initialize_profile(self.pressure_profile.nLayers,
                                                self.temperatureProfile,self.pressureProfile,
                                                None)
            self.compute_altitude_gravity_scaleheight_profile(self._inital_mu.muProfile)
            self._initialized=True
        
        #Now initialize the gas profile
        self._gas_profile.initialize_profile(self.pressure_profile.nLayers,
                                                self.temperatureProfile,self.pressureProfile,
                                                self.altitude_profile)
        
        #Compute gravity scale height
        self.compute_altitude_gravity_scaleheight_profile()

    def collect_fitting_parameters(self):
        self.fitting_parameters = {}
        self.fitting_parameters.update(self._planet.fitting_parameters())
        if self._star is not None:
            self.fitting_parameters.update(self._star.fitting_parameters())
        self.fitting_parameters.update(self.pressure_profile.fitting_parameters())
        self.fitting_parameters.update(self._temperature_profile.fitting_parameters())
        self.fitting_parameters.update(self._gas_profile.fitting_parameters())

        for contrib in self.contribution_list:
            self.fitting_parameters.update(contrib.fitting_parameters())


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
            mu_profile=self._gas_profile.muProfile

        # build the altitude profile from the bottom up
        nlayers = self.pressure_profile.nLayers
        H = np.zeros(nlayers)
        g = np.zeros(nlayers)
        z = np.zeros(nlayers)


        g[0] = self._planet.gravity # surface gravity (0th layer)
        H[0] = (KBOLTZ*self.temperatureProfile[0])/(mu_profile[0]*g[0]) # scaleheight at the surface (0th layer)

        for i in range(1, nlayers):
            deltaz = (-1.)*H[i-1]*np.log(self.pressureProfile[i]/self.pressureProfile[i-1])
            z[i] = z[i-1] + deltaz # altitude at the i-th layer

            with np.errstate(over='ignore'):
                g[i] = self._planet.gravity_at_height(z[i]) # gravity at the i-th layer
            with np.errstate(divide='ignore'):
                H[i] = (KBOLTZ*self.temperatureProfile[i])/(mu_profile[i]*g[i])

        self.altitude_profile = z
        self.scaleheight_profile = H
        self.gravity_profile = g

    @property
    def pressureProfile(self):
        return self.pressure_profile.profile

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
        return self.pressure_profile.nLayers



    @property
    def nativeWavenumberGrid(self):
        from taurex.cache.opacitycache import OpacityCache
        wavenumbergrid = [OpacityCache()[gas].wavenumberGrid for gas in self._gas_profile.active_gases]

        current_grid = None
        for wn in wavenumbergrid:
            if current_grid is None:
                current_grid = wn
            if wn.shape[0] > current_grid.shape[0]:
                current_grid = wn
        
        return current_grid
                

                


    def model(self,wngrid,return_contrib=False):
        self.initialize_profiles()
        for contrib in self.contribution_list:
            contrib.prepare(self,wngrid)
        return self.path_integral(wngrid,return_contrib)

    def path_integral(self,wngrid,return_contrib):
        raise NotImplementedError
