import configobj
from taurex.log import Logger
from .factory import create_chemistry, create_instrument, \
     create_model, create_observation, create_optimizer, \
     create_planet, create_star, create_temperature_profile, \
     create_pressure_profile
from taurex.optimizer import Optimizer


class ParameterParser(Logger):

    def __init__(self):
        super().__init__('ParamParser')
        self._read = False

    def transform(self, section, key):
        val = section[key]
        newval = val
        if isinstance(val, list):
            try:
                newval = list(map(float, val))
            except Exception:
                pass
        elif isinstance(val, (str)):
            if val.lower() in ['true',  'yes', 'yeah', 'yup', 'certainly', 'uh-huh', ]:
                newval = True
            elif val.lower() in ['false',  'no', 'nope', 'no-way', 'hell-no', ]:
                newval = False
            else:
                try:
                    newval = float(val)
                except Exception:
                    pass
        section[key] = newval
        return newval

    def setup_globals(self):
        from taurex.cache import GlobalCache
        from taurex.cache import CIACache, OpacityCache
        config = self._raw_config.dict()

        if 'Global' in config:
            global_config = config['Global']
            try:
                OpacityCache().set_opacity_path(config['Global']['xsec_path'])
            except KeyError:
                self.warning('No xsec path set, opacities '
                             'cannot be used in model')
            try:

                OpacityCache().set_interpolation(config['Global']
                                                       ['xsec_interpolation'])
                self.info('Interpolation mode set '
                          'to {}'.format(config['Global']
                                               ['xsec_interpolation']))
            except KeyError:
                self.info('Interpolation mode set to linear')

            try:
                CIACache().set_cia_path(config['Global']['cia_path'])
            except KeyError:
                self.warning('No cia path set, cia cannot be used in model')

            try:
                OpacityCache().set_memory_mode(config['Global']
                                                     ['xsec_in_memory'])
            except KeyError:
                self.warning('Xsecs will be loaded in memory')

            try:
                OpacityCache().enable_radis(config['Global']['use_radis'])
            except KeyError:
                self.warning('Radis is disabled')

            try:
                wn_start, wn_end, wn_points = config['Global']['radis_grid']

                OpacityCache().set_radis_wavenumber(wn_start, wn_end,
                                                    wn_points)
            except KeyError:
                self.warning('Radis default grid will be used')

            try:

                extension_paths = config['Global']['extension_paths']
                if isinstance(extension_paths, str):
                    extension_paths = [extension_paths, ]

                from .classfactory import ClassFactory
                ClassFactory().set_extension_paths(paths=extension_paths)
            except KeyError:
                pass


            gc = GlobalCache()

            for key, value in global_config.items():
                gc[key] = value

    def read(self, filename):
        import os.path
        if not os.path.isfile(filename):
            raise Exception('Input file {} does not exist'.format(filename))
        self._raw_config = configobj.ConfigObj(filename)
        self.debug('Raw Config file is {}, filename '
                   'is {}'.format(self._raw_config, filename))
        self._raw_config.walk(self.transform)
        config = self._raw_config.dict()
        self.debug('Config file is {}, filename '
                   'is {}'.format(config, filename))

    def generate_lightcurve(self):
        config = self._raw_config.dict()
        if 'Lightcurve' in config:
            from taurex.model.lightcurve.lightcurve import LightCurveModel
            model = self.generate_model()
            lightcurvefile = config['Lightcurve']['lc_pickle']
            return LightCurveModel(model, lightcurvefile)
        else:
            raise KeyError

    def generate_appropriate_model(self, obs=None):

        try:
            return self.generate_lightcurve()
        except KeyError:
            pass
        return self.generate_model(obs=obs)

    def generate_instrument(self, binner=None):
        from taurex.binning import NativeBinner
        config = self._raw_config.dict()
        if 'Instrument' in config:
            inst_config = config['Instrument']

            try:
                num_obs = inst_config.pop('num_observations')
            except KeyError:
                num_obs = 1

            if 'instrument' in inst_config:
                _bin = binner or NativeBinner
                if inst_config['instrument'].lower() in \
                        ('snr', 'signalnoise', ):
                    inst = self.create_snr(_bin, inst_config)
                    return inst, num_obs
            inst = create_instrument(inst_config)
            return inst, num_obs
        else:
            return None

    def create_snr(self, binner, config):
        from taurex.instruments.snr import SNRInstrument
        if binner is None:
            self.critical('Binning must be defined for SNR instrument')
            raise ValueError('Binning must be defined for SNR instrument')
        else:
            SNR = 10
            if 'SNR' in config:
                SNR = config['SNR']

            return SNRInstrument(SNR=SNR, binner=binner)

    def generate_optimizer(self):
        config = self._raw_config.dict()
        if 'Optimizer' in config:
            return create_optimizer(config['Optimizer'])
        else:
            None

    def setup_optimizer(self, optimizer: Optimizer):
        """
        Setup fitting parameters for optimizer
        """
        fitting_parameters = self.generate_fitting_parameters()

        self.info('Setting up optimizer')

        for key, value in fitting_parameters.items():
            fit = value['fit']
            bounds = value['bounds']
            mode = value['mode']
            factor = value['factor']
            prior = value['prior']

            if fit:
                self.info('Fitting: %s', key)
                optimizer.enable_fit(key)
            else:
                optimizer.disable_fit(key)

            if factor:
                optimizer.set_factor_boundary(key, factor)

            if bounds:
                optimizer.set_boundary(key, bounds)

            if mode:
                optimizer.set_mode(key, mode.lower())

            if prior is not None:
                optimizer.set_prior(key, prior)

        fitting_parameters = self.generate_derived_parameters()

        for key, value in fitting_parameters.items():
            compute = value['compute']

            if compute is not None:
                if compute:
                    self.info('Deriving %s', key)
                    optimizer.enable_derived(key)
                else:
                    optimizer.disable_derived(key)

    def generate_observation(self):

        config = self._raw_config.dict()
        if 'Observation' in config:
            observation_config = config['Observation']
            if 'lightcurve' in observation_config:
                from taurex.data.spectrum.lightcurve import ObservedLightCurve
                return ObservedLightCurve(observation_config['lightcurve'])

            elif 'observed_spectrum' in observation_config:
                from taurex.data.spectrum.observed import ObservedSpectrum
                return ObservedSpectrum(
                    observation_config['observed_spectrum'])

            elif 'taurex_spectrum' in observation_config:
                if observation_config['taurex_spectrum'] == 'self':
                    return 'self'
                from taurex.data.spectrum.taurex import TaurexSpectrum
                return TaurexSpectrum(observation_config['taurex_spectrum'])
            elif 'iraclis_spectrum' in observation_config:
                from taurex.data.spectrum.iraclis import IraclisSpectrum
                return IraclisSpectrum(observation_config['iraclis_spectrum'])
            else:
                config = self._raw_config.dict()
                return create_observation(observation_config)
        return None

    def create_manual_binning(self, config):
        import numpy as np
        import math
        from taurex.binning import FluxBinner, SimpleBinner

        binning_class = SimpleBinner

        if 'accurate' in config:
            if config['accurate']:
                binning_class = FluxBinner

        # Handle wavelength grid
        wngrid = None
        if 'wavelength_grid' in config:
            start, end, size = config['wavelength_grid']
            wngrid = 10000/np.linspace(start, end, int(size))
            wngrid = np.sort(wngrid)

        elif 'wavenumber_grid' in config:
            start, end, size = config['wavenumber_grid']
            wngrid = np.linspace(start, end, int(size))

        elif 'log_wavenumber_grid' in config:
            start, end, size = config['log_wavenumber_grid']
            start = math.log10(start)
            end = math.log10(end)
            wngrid = np.logspace(start, end, int(size))

        elif 'log_wavelength_grid' in config:
            start, end, size = config['log_wavelength_grid']
            start = math.log10(start)
            end = math.log10(end)
            wngrid = np.sort(10000/np.logspace(start, end, int(size)))
        elif 'wavelength_res' in config:
            from taurex.util.util import create_grid_res
            start, end, res = config['wavelength_res']
            wlgrid = create_grid_res(res, start, end)[:, 0].flatten()
            wngrid = 10000/wlgrid[::-1]

        if wngrid is None:
            self._logger.error('manual was selected and no grid was given.'
                               '(Use wavelength_grid, wavenumber_grid or '
                               'log versions)')
            raise Exception('manual selected but no grid given')

        return binning_class(wngrid), wngrid

    def generate_binning(self):

        config = self._raw_config.dict()
        if 'Binning' in config:
            binning_config = config['Binning']
            if 'bin_type' in binning_config:
                bin_type = binning_config['bin_type'].lower()

                if bin_type == 'native':
                    return 'native'
                elif bin_type == 'observed':
                    return 'observed'
                elif bin_type == 'manual':
                    return self.create_manual_binning(binning_config)
            else:
                return None
        return None

    def generate_model(self, chemistry=None, pressure=None,
                       temperature=None, planet=None,
                       star=None, obs=None):
        config = self._raw_config.dict()
        if 'Model' in config:
            if chemistry is None:
                chemistry = self.generate_chemistry_profile()
            if pressure is None:
                pressure = self.generate_pressure_profile()
            if temperature is None:
                temperature = self.generate_temperature_profile()
            if planet is None:
                planet = self.generate_planet()
            if star is None:
                star = self.generate_star()
            model = create_model(config['Model'], chemistry,
                                 temperature, pressure, planet, star, observation=obs)
        else:
            return None

        return model

    def generate_chemistry_profile(self):
        config = self._raw_config.dict()
        if 'Chemistry' in config:
            return create_chemistry(config['Chemistry'])
        else:
            return None

    def generate_pressure_profile(self):
        config = self._raw_config.dict()
        if 'Pressure' in config:
            return create_pressure_profile(config['Pressure'])
        else:
            return None

    def generate_temperature_profile(self):
        config = self._raw_config.dict()
        if 'Temperature' in config:
            return create_temperature_profile(config['Temperature'])
        else:
            return None

    def generate_planet(self):
        config = self._raw_config.dict()

        if 'Planet' in config:
            return create_planet(config['Planet'])
        else:
            return None

    def generate_star(self):
        config = self._raw_config.dict()

        if 'Star' in config:
            return create_star(config['Star'])
        else:
            return None

    def generate_fitting_parameters(self):
        from .factory import create_prior
        config = self._raw_config.dict()
        if 'Fitting' in config:
            fitting_config = config['Fitting']

            fitting_params = {}

            for key, value in fitting_config.items():
                fit_param, fit_type = key.split(':')
                if fit_param not in fitting_params:
                    fitting_params[fit_param] = {'fit': False,
                                                 'bounds': None,
                                                 'mode': None,
                                                 'factor': None,
                                                 'prior': None}

                if fit_type == 'prior':
                    value = create_prior(value)
                fitting_params[fit_param][fit_type] = value

            return fitting_params
        else:
            return {}

    def generate_derived_parameters(self):
        config = self._raw_config.dict()
        if 'Derive' in config:
            fitting_config = config['Derive']

            fitting_params = {}

            for key, value in fitting_config.items():
                fit_param, fit_type = key.split(':')
                if fit_param not in fitting_params:
                    fitting_params[fit_param] = {'compute': None}
                fitting_params[fit_param][fit_type] = value

            return fitting_params
        else:
            return {}
