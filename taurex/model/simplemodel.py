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
                            opacities=None,
                            cia=None,
                            opacity_path=None,
                            cia_path=None,
                            nlayers=100,
                            atm_min_pressure=1e-4,
                            atm_max_pressure=1e6,

                            ):
        super().__init__(name,opacities,cia,opacity_path,cia_path)
        


        self._planet = planet
        self._star=star
        self._pressure_profile = pressure_profile
        self._temperature_profile = temperature_profile
        self._gas_profile = gas_profile

        self.altitude_profile=None
        self.scaleheight_profile=None
        self.gravity_profile=None
        self.setup_defaults(nlayers,atm_min_pressure,atm_max_pressure)

        self._initialized = False

        self._sigma_opacities = None
        self._sigma_cia = None

    def _compute_inital_mu(self):
        from taurex.data.profiles.gas import ConstantGasProfile
        self._inital_mu=ConstantGasProfile()

    def setup_opacities_and_sigmas(self):
        self.load_opacities(molecule_filter=self._gas_profile.activeGases)
        self.load_cia()


        



    def setup_defaults(self,nlayers,atm_min_pressure,atm_max_pressure):

        if self._pressure_profile is None:
            from taurex.data.profiles.pressure import SimplePressureProfile
            self.info('No pressure profile defined, using simple pressure profile with')
            self.info('parameters nlayers: {}, atm_pressure_range=({},{})'.format(nlayers,atm_min_pressure,atm_max_pressure))
            self._pressure_profile = SimplePressureProfile(nlayers,atm_min_pressure,atm_max_pressure)

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
        
        self._pressure_profile.compute_pressure_profile()
        
        self._temperature_profile.initialize_profile(self._planet,
                    self._pressure_profile.nLayers,
                    self._pressure_profile.profile)
        
        #Initialize the atmosphere with a constant gas profile
        if self._initialized is False:
            self._inital_mu.initialize_profile(self._pressure_profile.nLayers,
                                                self.temperatureProfile,self.pressureProfile,
                                                None)
            self.compute_altitude_gravity_scaleheight_profile(self._inital_mu.muProfile)
            self._initialized=True
        
        #Now initialize the gas profile
        self._gas_profile.initialize_profile(self._pressure_profile.nLayers,
                                                self.temperatureProfile,self.pressureProfile,
                                                self.altitude_profile)
        
        #Compute gravity scale height
        self.compute_altitude_gravity_scaleheight_profile()

    def collect_fitting_parameters(self):
        self.fitting_parameters = {}
        self.fitting_parameters.update(self._planet.fitting_parameters())
        if self._star is not None:
            self.fitting_parameters.update(self._star.fitting_parameters())
        self.fitting_parameters.update(self._pressure_profile.fitting_parameters())
        self.fitting_parameters.update(self._temperature_profile.fitting_parameters())
        self.fitting_parameters.update(self._gas_profile.fitting_parameters())



    def build(self):
        self.info('Building model........')
        self._compute_inital_mu()
        self.collect_fitting_parameters()
        self.setup_opacities_and_sigmas()
        self.initialize_profiles()
        self.info('.....done!!!')




    # altitude, gravity and scale height profile
    def compute_altitude_gravity_scaleheight_profile(self,mu_profile=None):
        from taurex.constants import KBOLTZ
        if mu_profile is None:
            mu_profile=self._gas_profile.muProfile

        # build the altitude profile from the bottom up
        nlayers = self._pressure_profile.nLayers
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
        return self._pressure_profile.profile

    @property
    def temperatureProfile(self):
        return self._temperature_profile.profile

    @property
    def densityProfile(self):
        from taurex.constants import KBOLTZ
        return (self.pressureProfile)/(KBOLTZ*self.temperatureProfile)


    @property
    def nLayers(self):
        return self._pressure_profile.nLayers

    def model_opacities(self,wngrid):
        ngases = len(self._gas_profile.activeGases)

        self.sigma_xsec = np.zeros(self._pressure_profile.nLayers,ngases,
                len(wngrid))
        
        for idx_gas,gas in enumerate(self._gas_profile.activeGases):
            self.info('Recomputing active gas {} opacity'.format(gas))
            for idx_layer,temperature,pressure in enumerate(zip(self.temperatureProfile,self.pressureProfile)):
                self.sigma_xsec[idx_layer,idx_gas] = self.opacity_dict[gas].opacity(temperature,pressure,wngrid)


        return self.sigma_xsec

    
    def model_cia(self,wngrid):
        total_cia = len(self.cia_dict)
        if total_cia == 0:
            return
        self.sigma_cia = np.zeros(self._pressure_profile.nLayers,total_cia,
                len(wngrid))

        for cia_idx,cia in enumerate(self.cia_dict.values()):
            for idx_layer,temperature in enumerate(self.temperatureProfile):
                self.sigma_cia[idx_layer,cia_idx] = cia.cia(temperature,wngrid)



    def model(self,wngrid):
        self.initialize_profiles()
        self.model_opacities(wngrid)
        self.model_cia(wngrid)
        return self.path_integral()

    def path_integral(self):
        raise NotImplementedError
