from taurex.core import Singleton
from taurex.log import Logger
import inspect
import pkg_resources


class ClassFactory(Singleton):
    """
    A factory the discovers new
    classes from plugins
    """

    def init(self):
        self.log = Logger('ClassFactory')

        self.extension_paths = []
        self.reload_plugins()

    def set_extension_paths(self, paths=None, reload=True):
        self.extension_paths = paths
        if reload:
            self.reload_plugins()

    def reload_plugins(self):
        self.log.info('Reloading all modules and plugins')
        self.setup_batteries_included()
        self.setup_batteries_included_mixin()
        self.load_plugins()
        self.load_extension_paths()

    def setup_batteries_included_mixin(self):
        """
        Collect all the classes that are built into
        TauREx 3
        """
        from taurex.mixin import mixins

        self._temp_mixin_klasses = set()
        self._chem_mixin_klasses = set()
        self._gas_mixin_klasses = set()
        self._press_mixin_klasses = set()
        self._planet_mixin_klasses = set()
        self._star_mixin_klasses = set()
        self._inst_mixin_klasses = set()
        self._model_mixin_klasses = set()
        self._contrib_mixin_klasses = set()
        self._opt_mixin_klasses = set()
        self._obs_mixin_klasses = set()

        self._temp_mixin_klasses.update(
            self._collect_temperatures_mixin(mixins))
        self._chem_mixin_klasses.update(self._collect_chemistry_mixin(mixins))
        self._gas_mixin_klasses.update(self._collect_gas_mixin(mixins))
        self._press_mixin_klasses.update(self._collect_pressure_mixin(mixins))
        self._planet_mixin_klasses.update(self._collect_planets_mixin(mixins))
        self._star_mixin_klasses.update(self._collect_star_mixin(mixins))
        self._inst_mixin_klasses.update(self._collect_instrument_mixin(mixins))
        self._model_mixin_klasses.update(self._collect_model_mixin(mixins))
        self._obs_mixin_klasses.update(self._collect_observation_mixin(mixins))
        self._contrib_mixin_klasses.update(
            self._collect_contributions_mixin(mixins))

        self._opt_mixin_klasses.update(self._collect_optimizer_mixin(mixins))

    def setup_batteries_included(self):
        """
        Collect all the classes that are built into
        TauREx 3
        """
        from taurex import temperature, chemistry, pressure, planet, \
            stellar, instruments, model, contributions, optimizer, opacity, \
            spectrum
        from taurex.opacity import ktables
        from taurex.core import priors

        self._temp_klasses = set()
        self._chem_klasses = set()
        self._gas_klasses = set()
        self._press_klasses = set()
        self._planet_klasses = set()
        self._star_klasses = set()
        self._inst_klasses = set()
        self._model_klasses = set()
        self._contrib_klasses = set()
        self._opt_klasses = set()
        self._opac_klasses = set()
        self._ktab_klasses = set()
        self._obs_klasses = set()
        self._prior_klasses = set()

        self._temp_klasses.update(self._collect_temperatures(temperature))
        self._chem_klasses.update(self._collect_chemistry(chemistry))
        self._gas_klasses.update(self._collect_gas(chemistry))
        self._press_klasses.update(self._collect_pressure(pressure))
        self._planet_klasses.update(self._collect_planets(planet))
        self._star_klasses.update(self._collect_star(stellar))
        self._inst_klasses.update(self._collect_instrument(instruments))
        self._model_klasses.update(self._collect_model(model))
        self._obs_klasses.update(self._collect_observation(spectrum))
        self._contrib_klasses.update(
            self._collect_contributions(contributions))

        self._opt_klasses.update(self._collect_optimizer(optimizer))
        self._opac_klasses.update(self._collect_opacity(opacity))
        self._ktab_klasses.update(self._collect_ktables(ktables))
        self._prior_klasses.update(self._collect_priors(priors))

    def load_plugin(self, plugin_module):

        self._temp_klasses.update(self._collect_temperatures(plugin_module))
        self._chem_klasses.update(self._collect_chemistry(plugin_module))
        self._gas_klasses.update(self._collect_gas(plugin_module))
        self._press_klasses.update(self._collect_pressure(plugin_module))
        self._planet_klasses.update(self._collect_planets(plugin_module))
        self._star_klasses.update(self._collect_star(plugin_module))
        self._inst_klasses.update(self._collect_instrument(plugin_module))
        self._model_klasses.update(self._collect_model(plugin_module))
        self._obs_klasses.update(self._collect_observation(plugin_module))
        self._contrib_klasses.update(
            self._collect_contributions(plugin_module))
        self._opt_klasses.update(self._collect_optimizer(plugin_module))
        self._opac_klasses.update(self._collect_opacity(plugin_module))
        self._prior_klasses.update(self._collect_priors(plugin_module))
        self._ktab_klasses.update(self._collect_ktables(plugin_module))

        # Load any mixins

        self._temp_mixin_klasses.update(
            self._collect_temperatures_mixin(plugin_module))
        self._chem_mixin_klasses.update(
            self._collect_chemistry_mixin(plugin_module))
        self._gas_mixin_klasses.update(
            self._collect_gas_mixin(plugin_module))
        self._press_mixin_klasses.update(
            self._collect_pressure_mixin(plugin_module))
        self._planet_mixin_klasses.update(
            self._collect_planets_mixin(plugin_module))
        self._star_mixin_klasses.update(
            self._collect_star_mixin(plugin_module))
        self._inst_mixin_klasses.update(
            self._collect_instrument_mixin(plugin_module))
        self._model_mixin_klasses.update(
            self._collect_model_mixin(plugin_module))
        self._obs_mixin_klasses.update(
            self._collect_observation_mixin(plugin_module))
        self._contrib_mixin_klasses.update(
            self._collect_contributions_mixin(plugin_module))

        self._opt_mixin_klasses.update(
            self._collect_optimizer_mixin(plugin_module))

    def discover_plugins(self):
        plugins = {}
        failed_plugins = {}
        for entry_point in pkg_resources.iter_entry_points('taurex.plugins'):

            entry_point_name = entry_point.name

            try:
                module = entry_point.load()
            except Exception as e:
                # For whatever reason do not attempt to load the plugin
                self.log.warning('Could not load plugin %s', entry_point_name)
                self.log.warning('Reason: %s', str(e))
                failed_plugins[entry_point_name] = str(e)
                continue

            plugins[entry_point_name] = module

        return plugins, failed_plugins

    def load_plugins(self):
        plugins, failed_plugins = self.discover_plugins()
        self.log.info('----------Plugin loading---------')
        self.log.debug('Discovered plugins %s', plugins.values())

        for k, v in plugins.items():
            self.log.info('Loading %s', k)
            self.load_plugin(v)

    def load_extension_paths(self):
        import glob
        import os
        import pathlib
        import importlib
        paths = self.extension_paths
        if paths:
            # Make sure they're unique
            all_files = set(sum([glob.glob(
                                os.path.join(os.path.abspath(p), '*.py'))
                             for p in paths], []))

            for f in all_files:
                self.info('Loading extensions from %s', f)
                module_name = pathlib.Path(f).stem
                spec = importlib.util.spec_from_file_location(module_name, f)
                foo = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(foo)
                    self.load_plugin(foo)
                except Exception as e:
                    self.log.warning('Could not load extension from file %s',
                                     f)
                    self.log.warning('Reason: %s', str(e))

    def _collect_classes(self, module, base_klass):
        """
        Collects all classes that are a subclass of base
        """
        klasses = []
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for _, c in clsmembers:
            if issubclass(c, base_klass) and (c is not base_klass):
                self.log.debug(f' Found class {c.__name__}')
                klasses.append(c)

        return klasses

    def _collect_temperatures(self, module):
        from taurex.temperature import TemperatureProfile
        return self._collect_classes(module, TemperatureProfile)

    def _collect_chemistry(self, module):
        from taurex.chemistry import Chemistry
        return self._collect_classes(module, Chemistry)

    def _collect_gas(self, module):
        from taurex.chemistry import Gas
        return self._collect_classes(module, Gas)

    def _collect_pressure(self, module):
        from taurex.pressure import PressureProfile
        return self._collect_classes(module, PressureProfile)

    def _collect_planets(self, module):
        from taurex.planet import BasePlanet
        return self._collect_classes(module, BasePlanet)

    def _collect_star(self, module):
        from taurex.stellar import Star
        return self._collect_classes(module, Star)

    def _collect_instrument(self, module):
        from taurex.instruments import Instrument
        return self._collect_classes(module, Instrument)

    def _collect_model(self, module):
        from taurex.model import ForwardModel, SimpleForwardModel
        return [c for c in self._collect_classes(module, ForwardModel)
                if c is not SimpleForwardModel]

    def _collect_contributions(self, module):
        from taurex.contributions import Contribution
        return self._collect_classes(module, Contribution)

    def _collect_optimizer(self, module):
        from taurex.optimizer import Optimizer
        return self._collect_classes(module, Optimizer)

    def _collect_opacity(self, module):
        from taurex.opacity import Opacity, InterpolatingOpacity
        from taurex.opacity.ktables import KTable
        return [c for c in self._collect_classes(module, Opacity)
                if c is not InterpolatingOpacity and not issubclass(c, KTable)]

    def _collect_ktables(self, module):
        from taurex.opacity.ktables import KTable
        return [c for c in self._collect_classes(module, KTable)]

    def _collect_observation(self, module):
        from taurex.spectrum import BaseSpectrum
        return [c for c in self._collect_classes(module, BaseSpectrum)]

    def _collect_priors(self, module):
        from taurex.core.priors import Prior
        return [c for c in self._collect_classes(module, Prior)]

    # Mixins
    def _collect_temperatures_mixin(self, module):
        from taurex.mixin import TemperatureMixin
        return self._collect_classes(module, TemperatureMixin)

    def _collect_chemistry_mixin(self, module):
        from taurex.mixin import ChemistryMixin
        return self._collect_classes(module, ChemistryMixin)

    def _collect_gas_mixin(self, module):
        from taurex.mixin import GasMixin
        return self._collect_classes(module, GasMixin)

    def _collect_pressure_mixin(self, module):
        from taurex.mixin import PressureMixin
        return self._collect_classes(module, PressureMixin)

    def _collect_planets_mixin(self, module):
        from taurex.mixin import PlanetMixin
        return self._collect_classes(module, PlanetMixin)

    def _collect_star_mixin(self, module):
        from taurex.mixin import StarMixin
        return self._collect_classes(module, StarMixin)

    def _collect_instrument_mixin(self, module):
        from taurex.mixin import InstrumentMixin
        return self._collect_classes(module, InstrumentMixin)

    def _collect_model_mixin(self, module):
        from taurex.mixin import ForwardModelMixin
        return self._collect_classes(module, ForwardModelMixin)

    def _collect_contributions_mixin(self, module):
        from taurex.mixin import ContributionMixin
        return self._collect_classes(module, ContributionMixin)

    def _collect_optimizer_mixin(self, module):
        from taurex.mixin import OptimizerMixin
        return self._collect_classes(module, OptimizerMixin)

    def _collect_observation_mixin(self, module):
        from taurex.mixin import ObservationMixin
        return self._collect_classes(module, ObservationMixin)

    def list_from_base(self, klass_type):

        from taurex.temperature import TemperatureProfile
        from taurex.chemistry import Chemistry
        from taurex.chemistry import Gas
        from taurex.pressure import PressureProfile
        from taurex.planet import BasePlanet
        from taurex.stellar import Star
        from taurex.instruments import Instrument
        from taurex.model import ForwardModel
        from taurex.contributions import Contribution
        from taurex.optimizer import Optimizer
        from taurex.opacity import Opacity

        from taurex.opacity.ktables import KTable
        from taurex.spectrum import BaseSpectrum
        from taurex.core.priors import Prior

        klass_dict = {
            TemperatureProfile: self.temperatureKlasses,
            Chemistry: self.chemistryKlasses,
            Gas: self.gasKlasses,
            PressureProfile: self.pressureKlasses,
            BasePlanet: self.planetKlasses,
            Star: self.starKlasses,
            Instrument: self.instrumentKlasses,
            ForwardModel: self.modelKlasses,
            Contribution: self.contributionKlasses,
            Optimizer: self.optimizerKlasses,
            Opacity: self.opacityKlasses,
            KTable: self.ktableKlasses,
            BaseSpectrum: self.observationKlasses,
            Prior: self.priorKlasses,

        }

        return klass_dict[klass_type]

    @property
    def temperatureKlasses(self):
        return self._temp_klasses

    @property
    def chemistryKlasses(self):
        return self._chem_klasses

    @property
    def gasKlasses(self):
        return self._gas_klasses

    @property
    def pressureKlasses(self):
        return self._press_klasses

    @property
    def planetKlasses(self):
        return self._planet_klasses

    @property
    def starKlasses(self):
        return self._star_klasses

    @property
    def instrumentKlasses(self):
        return self._inst_klasses

    @property
    def modelKlasses(self):
        return self._model_klasses

    @property
    def contributionKlasses(self):
        return self._contrib_klasses

    @property
    def optimizerKlasses(self):
        return self._opt_klasses

    @property
    def opacityKlasses(self):
        return self._opac_klasses

    @property
    def ktableKlasses(self):
        return self._ktab_klasses

    @property
    def observationKlasses(self):
        return self._obs_klasses

    @property
    def priorKlasses(self):
        return self._prior_klasses

    # Mixins

    @property
    def temperatureMixinKlasses(self):
        return self._temp_mixin_klasses

    @property
    def chemistryMixinKlasses(self):
        return self._chem_mixin_klasses

    @property
    def gasMixinKlasses(self):
        return self._gas_mixin_klasses

    @property
    def pressureMixinKlasses(self):
        return self._press_mixin_klasses

    @property
    def planetMixinKlasses(self):
        return self._planet_mixin_klasses

    @property
    def starMixinKlasses(self):
        return self._star_mixin_klasses

    @property
    def instrumentMixinKlasses(self):
        return self._inst_mixin_klasses

    @property
    def modelMixinKlasses(self):
        return self._model_mixin_klasses

    @property
    def contributionMixinKlasses(self):
        return self._contrib_mixin_klasses

    @property
    def optimizerMixinKlasses(self):
        return self._opt_mixin_klasses

    @property
    def observationMixinKlasses(self):
        return self._obs_mixin_klasses
