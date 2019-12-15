from taurex.core import Singleton
from taurex.log import Logger
import inspect
# import pkg_resources

# discovered_plugins = {
#     entry_point.name: entry_point.load()
#     for entry_point
#     in pkg_resources.iter_entry_points('taurex.plugins')
# }


class ClassFactory(Singleton):
    """
    A factory the discovers new
    classes from plugins
    """
    def init(self):
        self.log = Logger('ClassFactory')

        self._temp_klasses = set()
        self._chem_klasses = set()
        self._gas_klasses = set()
        self._press_klasses = set()
    
    def setup_batteries_included(self):
        """
        Collect all the classes that are built into
        TauREx 3
        """
        from taurex import temperature, chemistry, pressure
        
        self._temp_klasses.update(self._collect_temperatures(temperature))
        self._chem_klasses.update(self._collect_chemistry(chemistry))
        self._gas_klasses.update(self._collect_gas(chemistry))
        self._press_klasses.update(self._collect_pressure(pressure))

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
