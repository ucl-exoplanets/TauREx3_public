import numpy as np
from .simplemodel import SimpleForwardModel
from taurex.constants import PI
from taurex.util.emission import black_body


class EmissionModel(SimpleForwardModel):
    """
    A forward model for eclipses

    Parameters
    ----------

    planet: :class:`~taurex.data.planet.Planet`, optional
        Planet model, default planet is Jupiter

    star: :class:`~taurex.data.stellar.star.Star`, optional
        Star model, default star is Sun-like

    pressure_profile: :class:`~taurex.data.profiles.pressure.pressureprofile.PressureProfile`, optional
        Pressure model, alternative is to set ``nlayers``, ``atm_min_pressure``
        and ``atm_max_pressure``

    temperature_profile: :class:`~taurex.data.profiles.temperature.tprofile.TemperatureProfile`, optional
        Temperature model, default is an :class:`~taurex.data.profiles.temperature.isothermal.Isothermal`
        profile at 1500 K

    chemistry: :class:`~taurex.data.profiles.chemistry.chemistry.Chemistry`, optional
        Chemistry model, default is
        :class:`~taurex.data.profiles.chemistry.taurexchemistry.TaurexChemistry` with
        ``H2O`` and ``CH4``

    nlayers: int, optional
        Number of layers. Used if ``pressure_profile`` is not defined.

    atm_min_pressure: float, optional
        Pressure at TOA. Used if ``pressure_profile`` is not defined.

    atm_max_pressure: float, optional
        Pressure at BOA. Used if ``pressure_profile`` is not defined.

    ngauss: int, optional
        Number of gaussian quadrature points, default = 4

    """

    def __init__(self,
                 planet=None,
                 star=None,
                 pressure_profile=None,
                 temperature_profile=None,
                 chemistry=None,
                 nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6,
                 ngauss=4,
                 ):
        super().__init__(self.__class__.__name__,
                         planet,
                         star,
                         pressure_profile,
                         temperature_profile,
                         chemistry,
                         nlayers,
                         atm_min_pressure,
                         atm_max_pressure)

        self.set_num_gauss(ngauss)
        self._clamp = 10

    def set_num_gauss(self,value,coeffs=None):
        self._ngauss = int(value)

        mu,weight =np.polynomial.legendre.leggauss(self._ngauss*2)
        self._mu_quads =mu[self._ngauss:] 
        self._wi_quads =weight[self._ngauss:]
        self._coeffs = coeffs
        if coeffs is None: 
            self._coeffs = np.ones(self._ngauss)

            

    def set_quadratures(self,mu,weight,coeffs=None):
        self._mu_quads =mu
        self._wi_quads =weight
        self._coeffs = coeffs
        if coeffs is None: 
            self._coeffs = np.ones(self._ngauss)

    def compute_final_flux(self, f_total):
        star_sed = self._star.spectralEmissionDensity

        self.debug('Star SED: %s', star_sed)
        # quit()
        star_radius = self._star.radius
        planet_radius = self._planet.fullRadius
        self.debug('star_radius %s', self._star.radius)
        self.debug('planet_radius %s', self._star.radius)
        last_flux = (f_total/star_sed) * (planet_radius/star_radius)**2

        self.debug('last_flux %s', last_flux)

        return last_flux


    def partial_model(self,wngrid=None,cutoff_grid=True):
        from taurex.util.util import clip_native_to_wngrid
        self.initialize_profiles()

        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid,wngrid)
        self._star.initialize(native_grid)

        for contrib in self.contribution_list:
            
            contrib.prepare(self,native_grid)
            
        return self.evaluate_emission(native_grid,False)


    def evaluate_emission(self,wngrid,return_contrib):
        import numexpr as ne
        dz=np.gradient(self.altitudeProfile)
        

        density = self.densityProfile

        wngrid_size = wngrid.shape[0]

        total_layers = self.nLayers

        temperature = self.temperatureProfile
        tau = np.zeros(shape=(self.nLayers, wngrid_size))
        surface_tau = np.zeros(shape=(1, wngrid_size))

        layer_tau = np.zeros(shape=(1, wngrid_size))

        dtau = np.zeros(shape=(1, wngrid_size))

        # Do surface first
        # for layer in range(total_layers):
        for contrib in self.contribution_list:
            contrib.contribute(self, 0, total_layers, 0, 0,
                               density, surface_tau, path_length=dz)
        self.debug('density = %s', density[0])
        self.debug('surface_tau = %s', surface_tau)

        BB = black_body(wngrid, temperature[0])/PI

        _mu = 1.0/self._mu_quads[:, None]
        _w = self._wi_quads[:, None]
        I = BB * (np.exp(-surface_tau*_mu))

        self.debug('I1_pre %s', I)
        # Loop upwards
        for layer in range(total_layers):
            layer_tau[...] = 0.0
            dtau[...] = 0.0
            for contrib in self.contribution_list:
                contrib.contribute(self, layer+1, total_layers,
                                   0, 0, density, layer_tau, path_length=dz)
                contrib.contribute(self, layer, layer+1, 0,
                                   0, density, dtau, path_length=dz)



            dtau_calc = 0.0
            if dtau.min() < self._clamp:
                dtau_calc = np.exp(-dtau)
            layer_tau_calc = 0.0
            if layer_tau.min() < self._clamp:
                layer_tau_calc = np.exp(-layer_tau)

            _tau = layer_tau_calc - dtau_calc

            if isinstance(_tau,float):
                tau[layer] += _tau
            else:
                tau[layer] += _tau[0]
            # for contrib in self.contribution_list:

            self.debug('Layer_tau[%s]=%s', layer, layer_tau)

            dtau += layer_tau

            self.debug('dtau[%s]=%s', layer, dtau)
            BB = black_body(wngrid, temperature[layer])/PI
            self.debug('BB[%s]=%s,%s', layer, temperature[layer], BB)

            dtau_calc = 0.0
            if dtau.min() < self._clamp:
                dtau_calc = np.exp(-dtau*_mu)
            layer_tau_calc = 0.0
            if layer_tau.min() < self._clamp:
                layer_tau_calc = np.exp(-layer_tau*_mu)



            I += BB * (layer_tau_calc - dtau_calc)

        self.debug('I: %s', I)

        return I,_mu,_w,tau

    def path_integral(self,wngrid,return_contrib):

        I,_mu,_w,tau = self.evaluate_emission(wngrid,return_contrib)
        self.debug('I: %s',I)

        flux_total = 2.0*np.pi*sum(I*(_w/_mu))
        self.debug('flux_total %s',flux_total)
        
        return self.compute_final_flux(flux_total).ravel(),tau

    def write(self, output):
        model = super().write(output)
        model.write_scalar('ngauss', self._ngauss)
        return model

    @classmethod
    def input_keywords(self):
        return ['emission', 'eclipse', ]
