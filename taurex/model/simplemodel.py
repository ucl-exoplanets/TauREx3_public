import numpy as np
from .model import ForwardModel
from taurex.util.util import clip_native_to_wngrid


class SimpleForwardModel(ForwardModel):
    """ A 'simple' base model in the sense that its just
    a fairly standard single profiles model.
    It will handle settingup and initializing, building
    collecting parameters from given profiles for you.
    The only method that requires implementation is:

    - :func:`path_integral`

    Parameters
    ----------
    name: str
        Name to use in logging

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

    """

    def __init__(self, name,
                 planet=None,
                 star=None,
                 pressure_profile=None,
                 temperature_profile=None,
                 chemistry=None,
                 nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6):
        super().__init__(name)

        self._planet = planet
        self._star = star
        self._pressure_profile = pressure_profile
        self._temperature_profile = temperature_profile
        self._chemistry = chemistry
        self.debug('Passed: %s %s %s %s %s', planet, star, pressure_profile,
                   temperature_profile, chemistry)
        self.altitude_profile = None
        self.scaleheight_profile = None
        self.gravity_profile = None
        self._setup_defaults(nlayers, atm_min_pressure, atm_max_pressure)

        self._initialized = False

        self._sigma_opacities = None

        self._native_grid = None

    def _compute_inital_mu(self):
        from taurex.data.profiles.chemistry import TaurexChemistry, ConstantGas
        tc = TaurexChemistry()
        tc.addGas(ConstantGas('H2O'))
        self._inital_mu = tc

    def _setup_defaults(self, nlayers, atm_min_pressure, atm_max_pressure):

        if self._pressure_profile is None:
            from taurex.data.profiles.pressure import SimplePressureProfile
            self.info('No pressure profile defined, using simple pressure '
                      'profile with')
            self.info('parameters nlayers: %s, atm_pressure_range=(%s,%s)',
                      nlayers, atm_min_pressure, atm_max_pressure)

            self._pressure_profile = \
                SimplePressureProfile(nlayers, atm_min_pressure,
                                      atm_max_pressure)

        if self._planet is None:
            from taurex.data import Planet
            self.warning('No planet defined, using Jupiter as planet')
            self._planet = Planet()

        if self._temperature_profile is None:
            from taurex.data.profiles.temperature import Isothermal
            self.warning('No temeprature profile defined using default '
                         'Isothermal profile with T=1500 K')
            self._temperature_profile = Isothermal()

        if self._chemistry is None:
            from taurex.data.profiles.chemistry import TaurexChemistry, \
                ConstantGas
            tc = TaurexChemistry()
            self.warning('No gas profile set, using constant profile with H2O '
                         'and CH4')
            tc.addGas(ConstantGas('H2O', mix_ratio=1e-5))
            tc.addGas(ConstantGas('CH4', mix_ratio=1e-6))
            self._chemistry = tc

        if self._star is None:
            from taurex.data.stellar import BlackbodyStar
            self.warning('No star, using the Sun')
            self._star = BlackbodyStar()

    def initialize_profiles(self):
        """
        Initializes all profiles
        """

        self.info('Computing pressure profile')

        self.pressure.compute_pressure_profile()

        self._temperature_profile.initialize_profile(self._planet,
                                                     self.pressure.nLayers,
                                                     self.pressure.profile)

        # Initialize the atmosphere with a constant gas profile
        if self._initialized is False:
            self._inital_mu.initialize_chemistry(self.pressure.nLayers,
                                                 self.temperatureProfile,
                                                 self.pressureProfile,
                                                 None)

            self._compute_altitude_gravity_scaleheight_profile(
                self._inital_mu.muProfile)

            self._initialized = True

        # Now initialize the gas profile real
        self._chemistry.initialize_chemistry(self.pressure.nLayers,
                                             self.temperatureProfile,
                                             self.pressureProfile,
                                             self.altitude_profile)

        # Compute gravity scale height
        self._compute_altitude_gravity_scaleheight_profile()

    def collect_fitting_parameters(self):
        """
        Collects all fitting parameters from all
        profiles within the forward model
        """

        self._fitting_parameters = {}
        self._fitting_parameters.update(self.fitting_parameters())
        self._fitting_parameters.update(self._planet.fitting_parameters())
        if self._star is not None:
            self._fitting_parameters.update(self._star.fitting_parameters())
        self._fitting_parameters.update(self.pressure.fitting_parameters())

        self._fitting_parameters.update(
            self._temperature_profile.fitting_parameters())

        self._fitting_parameters.update(self._chemistry.fitting_parameters())

        for contrib in self.contribution_list:
            self._fitting_parameters.update(contrib.fitting_parameters())

        self.debug('Available Fitting params: %s',
                   list(self._fitting_parameters.keys()))

    def build(self):
        """
        Build the forward model. Must be called at least
        once before running :func:`model`
        """

        self.contribution_list.sort(key=lambda x: x.order)

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
    def _compute_altitude_gravity_scaleheight_profile(self, mu_profile=None):
        """
        Computes altitude, gravity and scale height of the atmosphere.
        Only call after :func:`build` has been called at least once.

        Parameters
        ----------
        mu_profile, optional:
            Molecular weight profile at each layer

        """

        from taurex.constants import KBOLTZ
        if mu_profile is None:
            mu_profile = self._chemistry.muProfile

        # build the altitude profile from the bottom up
        nlayers = self.pressure.nLayers
        H = np.zeros(nlayers)
        g = np.zeros(nlayers)
        z = np.zeros(nlayers)

        # surface gravity (0th layer)
        g[0] = self._planet.gravity
        # scaleheight at the surface (0th layer)
        H[0] = (KBOLTZ*self.temperatureProfile[0])/(mu_profile[0]*g[0])

        for i in range(1, nlayers):
            deltaz = (-1.)*H[i-1]*np.log(
                self.pressure.pressure_profile_levels[i] /
                self.pressure.pressure_profile_levels[i-1])

            z[i] = z[i-1] + deltaz  # altitude at the i-th layer

            with np.errstate(over='ignore'):
                # gravity at the i-th layer
                g[i] = self._planet.gravity_at_height(z[i])
                self.debug('G[%s] = %s', i, g[i])

            with np.errstate(divide='ignore'):
                H[i] = (KBOLTZ*self.temperatureProfile[i])/(mu_profile[i]*g[i])

        self.altitude_profile = z
        self.scaleheight_profile = H
        self.gravity_profile = g

    @property
    def pressureProfile(self):
        """
        Atmospheric pressure profile in Pa
        """
        return self.pressure.profile

    @property
    def temperatureProfile(self):
        """
        Atmospheric temperature profile in K
        """
        return self._temperature_profile.profile

    @property
    def densityProfile(self):
        """
        Atmospheric density profile in m-3
        """
        from taurex.constants import KBOLTZ
        return (self.pressureProfile)/(KBOLTZ*self.temperatureProfile)

    @property
    def altitudeProfile(self):
        """
        Atmospheric altitude profile in m
        """
        return self.altitude_profile

    @property
    def nLayers(self):
        """
        Number of layers
        """
        return self.pressure.nLayers

    @property
    def chemistry(self):
        """
        Chemistry model
        """
        return self._chemistry

    @property
    def pressure(self):
        """
        Pressure model
        """
        return self._pressure_profile

    @property
    def temperature(self):
        """
        Temperature model
        """
        return self._temperature_profile

    @property
    def star(self):
        """
        Star model
        """
        return self._star

    @property
    def planet(self):
        """
        Planet model
        """
        return self._planet

    @property
    def nativeWavenumberGrid(self):
        """

        Searches through active molecules to determine the
        native wavenumber grid

        Returns
        -------

        wngrid: :obj:`array`
            Native grid

        Raises
        ------
        InvalidModelException
            If no active molecules in atmosphere
        """
        from taurex.exceptions import InvalidModelException
        from taurex.cache.opacitycache import OpacityCache

        active_gases = self.chemistry.activeGases

        wavenumbergrid = \
            [OpacityCache()[gas].wavenumberGrid for gas in active_gases]

        current_grid = None
        for wn in wavenumbergrid:
            if current_grid is None:
                current_grid = wn
            if wn.shape[0] > current_grid.shape[0]:
                current_grid = wn

        if current_grid is None:
            self.error('No active molecules detected')
            self.error('Most likely no cross-sections were detected')
            raise InvalidModelException('No active absorbing molecules')

        return current_grid

    def model(self, wngrid=None, cutoff_grid=True):
        """
        Runs the forward model

        Parameters
        ----------

        wngrid: :obj:`array`, optional
            Wavenumber grid, default is to use native grid

        cutoff_grid: bool
            Run model only on ``wngrid`` given, default is ``True``

        Returns
        -------

        native_grid: :obj:`array`
            Native wavenumber grid, clipped if ``wngrid`` passed

        depth: :obj:`array`
            Resulting depth

        tau: :obj:`array`
            Optical depth.

        extra: ``None``
            Empty
        """

        # Setup profiles
        self.initialize_profiles()

        # Clip grid if necessary
        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)

        # Initialize star
        self._star.initialize(native_grid)

        # Prepare contributions
        for contrib in self.contribution_list:
            contrib.prepare(self, native_grid)

        # Compute path integral
        absorp, tau = self.path_integral(native_grid, False)

        return native_grid, absorp, tau, None

    def model_contrib(self, wngrid=None, cutoff_grid=True):
        """
        Models each contribution seperately
        """
        # Setup profiles
        self.initialize_profiles()

        # Copy over contribution list
        full_contrib_list = self.contribution_list
        # Get the native grid
        native_grid = self.nativeWavenumberGrid

        # Clip grid
        all_contrib_dict = {}
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)

        # Initialize star
        self._star.initialize(native_grid)

        for contrib in full_contrib_list:
            self.contribution_list = [contrib]
            contrib.prepare(self, native_grid)
            absorp, tau = self.path_integral(native_grid, False)
            all_contrib_dict[contrib.name] = (absorp, tau, None)

        self.contribution_list = full_contrib_list
        return native_grid, all_contrib_dict

    def model_full_contrib(self, wngrid=None, cutoff_grid=True):
        """

        Like :func:`model_contrib` except all components for
        each contribution are modelled

        """
        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)

        self.initialize_profiles()
        self._star.initialize(native_grid)

        result_dict = {}

        full_contrib_list = self.contribution_list

        self.debug('NATIVE GRID %s', native_grid.shape)

        self.info('Modelling each contribution.....')
        for contrib in full_contrib_list:
            self.contribution_list = [contrib]
            contrib_name = contrib.name
            contrib_res_list = []

            for name, __ in contrib.prepare_each(self, native_grid):
                self.info('\t%s---%s contribtuion', contrib_name, name)
                absorp, tau = self.path_integral(native_grid, False)
                contrib_res_list.append((name, absorp, tau, None))

            result_dict[contrib_name] = contrib_res_list

        self.contribution_list = full_contrib_list
        return native_grid, result_dict

    def compute_error(self, samples, wngrid=None, binner=None):
        """

        Computes standard deviations from samples

        Parameters
        ----------

        samples: 

        """
        from taurex.util.math import OnlineVariance
        tp_profiles = OnlineVariance()
        active_gases = OnlineVariance()
        inactive_gases = OnlineVariance()

        if binner is not None:
            binned_spectrum = OnlineVariance()
        else:
            binned_spectrum = None
        native_spectrum = OnlineVariance()

        for weight in samples():

            native_grid, native, tau, _ = self.model(wngrid=wngrid,
                                                     cutoff_grid=False)

            tp_profiles.update(self.temperatureProfile, weight=weight)
            active_gases.update(self.chemistry.activeGasMixProfile,
                                weight=weight)
            inactive_gases.update(self.chemistry.inactiveGasMixProfile,
                                  weight=weight)

            native_spectrum.update(native, weight=weight)

            if binned_spectrum is not None:
                binned = binner.bindown(native_grid, native)[1]
                binned_spectrum.update(binned, weight=weight)

        profile_dict = {}
        spectrum_dict = {}

        tp_std = np.sqrt(tp_profiles.parallelVariance())
        active_std = np.sqrt(active_gases.parallelVariance())
        inactive_std = np.sqrt(inactive_gases.parallelVariance())

        profile_dict['temp_profile_std'] = tp_std
        profile_dict['active_mix_profile_std'] = active_std
        profile_dict['inactive_mix_profile_std'] = inactive_std

        spectrum_dict['native_std'] = \
            np.sqrt(native_spectrum.parallelVariance())

        if binned_spectrum is not None:
            spectrum_dict['binned_std'] = \
                np.sqrt(binned_spectrum.parallelVariance())

        return profile_dict, spectrum_dict

    def path_integral(self, wngrid, return_contrib):
        raise NotImplementedError

    def write(self, output):
        # Run a model if needed
        self.model()

        model = super().write(output)

        self._chemistry.write(model)
        self._temperature_profile.write(model)
        self.pressure.write(model)
        self._planet.write(model)
        self._star.write(model)
        return model
