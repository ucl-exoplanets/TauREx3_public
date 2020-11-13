from ..core import Fittable


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


class ChemistryMixin(Mixin):
    pass


class ForwardModelMixin(Mixin):
    pass


class OptimizerMixin(Mixin):
    pass



def enhance_class(base_klass, mixins, **kwargs):

    if not hasattr(mixins, '__len__'):
        mixins = [mixins]

    all_classes = tuple(mixins) + tuple([base_klass])
    new_name = '-'.join([x.__name__[:10] for x in all_classes])

    new_klass = type(new_name, all_classes, {'__init__': mixed_init})

    return new_klass(**kwargs)


