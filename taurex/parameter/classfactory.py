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

        self.setup_batteries_included()

    def setup_batteries_included(self):
        """
        Collect all the classes that are built into
        TauREx 3
        """
        from taurex import temperature, chemistry, pressure, planet, \
            stellar, instruments, model, contributions, optimizer, opacity

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

        self._temp_klasses.update(self._collect_temperatures(temperature))
        self._chem_klasses.update(self._collect_chemistry(chemistry))
        self._gas_klasses.update(self._collect_gas(chemistry))
        self._press_klasses.update(self._collect_pressure(pressure))
        self._planet_klasses.update(self._collect_planets(planet))
        self._star_klasses.update(self._collect_star(stellar))
        self._inst_klasses.update(self._collect_instrument(instruments))
        self._model_klasses.update(self._collect_model(model))
        self._contrib_klasses.update(
            self._collect_contributions(contributions))

        self._opt_klasses.update(self._collect_optimizer(optimizer))
        self._opac_klasses.update(self._collect_opacity(opacity))

    def load_plugin(self, plugin_module):

        self._temp_klasses.update(self._collect_temperatures(plugin_module))
        self._chem_klasses.update(self._collect_chemistry(plugin_module))
        self._gas_klasses.update(self._collect_gas(plugin_module))
        self._press_klasses.update(self._collect_pressure(plugin_module))
        self._planet_klasses.update(self._collect_planets(plugin_module))
        self._star_klasses.update(self._collect_star(plugin_module))
        self._inst_klasses.update(self._collect_instrument(plugin_module))
        self._model_klasses.update(self._collect_model(plugin_module))
        self._contrib_klasses.update(
            self._collect_contributions(plugin_module))
        self._opt_klasses.update(self._collect_optimizer(plugin_module))
        self._opac_klasses.update(self._collect_opacity(plugin_module))

    def discover_plugins(self):
        return {
            entry_point.name: entry_point.load()
            for entry_point
            in pkg_resources.iter_entry_points('taurex.plugins')
        }

    def load_plugins(self):
        plugins = self.discover_plugins()
        self.log.info('----------Plugin loading---------')
        self.log.info('Discovered plugins %s', plugins.values())

        for k, v in plugins.items():
            self.log.info()

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
        from taurex.planet import Planet
        return self._collect_classes(module, Planet)

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
        return [c for c in self._collect_classes(module, Opacity)
                if c is not InterpolatingOpacity]

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
