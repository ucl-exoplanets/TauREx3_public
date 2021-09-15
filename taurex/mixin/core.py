from ..core import Fittable
from ..core import Citable
from ..parameter.factory import log


# Try and create __init_mixin__

def mixed_init(self, **kwargs):
    import inspect
    new_class = self.__class__
    base_class = self.__class__.__bases__[-1]
    args = inspect.getfullargspec(base_class.__init__).args[1:]
    new_kwargs = {}
    for k, v in kwargs.items():
        if k in args:
            new_kwargs[k] = v

    super(new_class, self).__init__(**new_kwargs)

    new_kwargs = {}

    for klass in reversed(new_class.__bases__[:-1]):
        args = inspect.getfullargspec(klass.__init_mixin__).args[1:]
        new_kwargs = {}
        for k, v in kwargs.items():
            if k in args:
                new_kwargs[k] = v
        klass.__init_mixin__(self, **new_kwargs)


class Mixin(Fittable, Citable):

    def __init__(self, **kwargs):
        old_fitting_parameters = {}
        old_derived_parameters = {}
        if hasattr(self, '_param_dict'):
            old_fitting_parameters = self._param_dict
            old_derived_parameters = self._derived_dict
        super().__init__(**kwargs)

        self._param_dict.update(old_fitting_parameters)
        self._derived_dict.update(old_derived_parameters)

    def __init__mixin(self):
        pass

    @classmethod
    def input_keywords(self):
        raise NotImplementedError


class StarMixin(Mixin):
    """
    A mixin that enhances :class:`~taurex.data.stellar.star.Star`
    """

    pass


class TemperatureMixin(Mixin):
    pass


class PlanetMixin(Mixin):
    pass


class ContributionMixin(Mixin):
    pass


class ChemistryMixin(Mixin):
    pass


class PressureMixin(Mixin):
    pass


class ForwardModelMixin(Mixin):
    pass


class ObservationMixin(Mixin):
    pass


class OptimizerMixin(Mixin):
    pass


class GasMixin(Mixin):
    pass


class InstrumentMixin(Mixin):
    pass


def determine_mixin_args(klasses):
    import inspect
    all_kwargs = []
    defaults = []
    for klass in klasses:
        argspec = inspect.getfullargspec(klass.__init__)
        if issubclass(klass, Mixin):
            argspec = inspect.getfullargspec(klass.__init_mixin__)
        if not argspec.defaults:
            continue

        args = argspec.args

        defaults.extend(argspec.defaults)
        num_defaults = len(argspec.defaults)
        all_kwargs.extend(args[-num_defaults:])
    all_kwargs = {k: v for k, v in zip(all_kwargs, defaults)}
    return all_kwargs


def build_new_mixed_class(base_klass, mixins):
    if not hasattr(mixins, '__len__'):
        mixins = [mixins]

    all_classes = tuple(mixins) + tuple([base_klass])
    new_name = '+'.join([x.__name__[:10] for x in all_classes])

    new_klass = type(new_name, all_classes, {'__init__': mixed_init})

    return new_klass


def enhance_class(base_klass, mixins, **kwargs):
    new_klass = build_new_mixed_class(base_klass, mixins)
    all_kwargs = determine_mixin_args(new_klass.__bases__)

    for k in kwargs:
        if k not in all_kwargs:
            log.error('Object {} does not have '
                      'parameter {}'.format(new_klass, k))
            log.error('Available parameters are %s', all_kwargs)
            raise KeyError(f'Object {new_klass} does not have parameter {k}')

    return new_klass(**kwargs)
