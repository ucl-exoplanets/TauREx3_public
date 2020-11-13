from ..core import Fittable
from ..parameter.factory import log

def mixed_init(self, **kwargs):
    import inspect
    for klass in reversed(self.__class__.__bases__):
        args = inspect.getfullargspec(klass.__init__).args
        new_kwargs = {}
        for k, v in kwargs.items():
            if k in args:
                new_kwargs[k] = v
        klass.__init__(self, **new_kwargs)


class Mixin(Fittable):

    def __init__(self):
        old_fitting_parameters = {}
        if hasattr(self, '_fitting_parameters'):
            old_fitting_parameters = self._fitting_parameters
        super().__init__()
            
        self._fitting_parameters.update(old_fitting_parameters)

    @classmethod
    def input_keywords(self):
        raise NotImplementedError


class StarMixin(Mixin):
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
        args = argspec.args
        defaults.extend(argspec.defaults)
        num_defaults = len(argspec.defaults)
        all_kwargs.extend(args[-num_defaults:])
    all_kwargs = {k: v for k, v in zip(all_kwargs, defaults)}
    return all_kwargs


def build_new_mixed_class(base_klass, mixins):
    all_classes = tuple(mixins) + tuple([base_klass])
    new_name = '+'.join([x.__name__[:10] for x in all_classes])
    new_klass = type(new_name, all_classes, {'__init__': mixed_init})
    return new_klass


def enhance_class(base_klass, mixins, **kwargs):
    if not hasattr(mixins, '__len__'):
        mixins = [mixins]

    all_classes = tuple(mixins) + tuple([base_klass])
    all_kwargs = determine_mixin_args(all_classes)

    new_name = '+'.join([x.__name__ for x in all_classes])

    new_klass = type(new_name, all_classes, {'__init__': mixed_init})

    for k in kwargs:
        if k not in all_kwargs:
            log.error('Object {} does not have parameter {}'.format(new_name, k))
            log.error('Available parameters are %s', all_kwargs)
            raise KeyError(f'Object {new_name} does not have parameter {k}')

    return new_klass(**kwargs)


