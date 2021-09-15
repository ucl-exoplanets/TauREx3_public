from taurex.log import Logger
from .classfactory import ClassFactory
log = Logger('Factory')


def get_keywordarg_dict(klass, is_mixin=False):

    import inspect
    init_dicts = {}
    if not is_mixin:
        init_dicts = {}
        args, varargs, varkw, defaults = inspect.getargspec(klass.__init__)
        log.debug('Inpection {} {} {} {}'.format(args,
                                                 varargs,
                                                 varkw,
                                                 defaults))
        if defaults is None:
            return init_dicts

        keyword_args = args[-len(defaults):]

        for keyword, value in zip(keyword_args, defaults):
            init_dicts[keyword] = value
    else:
        from taurex.mixin.core import determine_mixin_args
        init_dicts = determine_mixin_args(klass.__bases__)

    return init_dicts


def create_klass(config, klass, is_mixin):
    kwargs = get_keywordarg_dict(klass, is_mixin)

    for key in config:
        if key in kwargs:
            value = config[key]
            kwargs[key] = value
        else:
            log.error(
                'Object {} does not have parameter {}'.format(klass.__name__,
                                                              key))
            log.error('Available parameters are %s', kwargs.keys())
            raise KeyError
    obj = klass(**kwargs)
    # for key,value in config.items():
    #     try:
    #         obj[key] = value
    #     except KeyError:

    #         raise KeyError
    return obj


def mixin_factory(profile_type, baseclass):
    mixin_def = {
        'TemperatureProfile': 'temperatureMixinKlasses',
        'Chemistry': 'chemistryMixinKlasses',
        'Gas': 'gasMixinKlasses',
        'PressureProfile': 'pressureMixinKlasses',
        'Planet': 'planetMixinKlasses',
        'Star': 'starMixinKlasses',
        'Instrument': 'instrumentMixinKlasses',
        'ForwardModel': 'modelMixinKlasses',
        'Contribution': 'contributionMixinKlasses',
        'Optimizer': 'optimizerMixinKlasses',
        'BaseSpectrum': 'observationMixinKlasses',
    }
    cf = ClassFactory()

    attri_list = getattr(cf, mixin_def[baseclass.__name__])
    for klass in attri_list:
        try:
            if profile_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('{} {} not '
                              'implemented'.format(baseclass.__name__,
                                                   profile_type))


def generic_factory(profile_type, baseclass):

    mixin_def = {
        'TemperatureProfile': 'temperatureKlasses',
        'Chemistry': 'chemistryKlasses',
        'Gas': 'gasKlasses',
        'PressureProfile': 'pressureKlasses',
        'BasePlanet': 'planetKlasses',
        'Star': 'starKlasses',
        'Instrument': 'instrumentKlasses',
        'ForwardModel': 'modelKlasses',
        'Contribution': 'contributionKlasses',
        'Optimizer': 'optimizerKlasses',
        'BaseSpectrum': 'observationKlasses',
    }
    cf = ClassFactory()

    attri_list = getattr(cf, mixin_def[baseclass.__name__])
    for klass in attri_list:
        try:
            if profile_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('{} {} not '
                              'implemented'.format(baseclass.__name__,
                                                   profile_type))


def create_profile(config, factory, baseclass=None,
                   keyword_type='profile_type'):
    config, klass, mixin = \
        determine_klass(config, keyword_type,
                        lambda x: generic_factory(x, baseclass),
                        baseclass=baseclass)

    obj = create_klass(config, klass, mixin)

    return obj


def gas_factory(profile_type):
    cf = ClassFactory()

    for klass in cf.gasKlasses:
        try:
            if profile_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('Gas profile {} not '
                              'implemented'.format(profile_type))


def temp_factory(profile_type):
    cf = ClassFactory()

    for klass in cf.temperatureKlasses:
        try:
            if profile_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('Temperature profile {} not '
                              'implemented'.format(profile_type))


def chemistry_factory(profile_type):
    cf = ClassFactory()
    for klass in cf.chemistryKlasses:
        try:
            if profile_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('Chemistry {} not '
                              'implemented'.format(profile_type))


def pressure_factory(profile_type):
    cf = ClassFactory()
    for klass in cf.pressureKlasses:
        try:
            if profile_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('Pressure profile {} not '
                              'implemented'.format(profile_type))


def star_factory(star_type):
    cf = ClassFactory()
    for klass in cf.starKlasses:
        try:
            if star_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('Star of type {} not '
                              'implemented'.format(star_type))


def create_gas_profile(config):
    from taurex.data.profiles.chemistry.gas import Gas
    return create_profile(config, gas_factory, Gas)


def create_temperature_profile(config):
    from taurex.data.profiles.temperature import TemperatureProfile
    return create_profile(config, temp_factory, TemperatureProfile)


def create_pressure_profile(config):
    from taurex.data.profiles.pressure.pressureprofile import PressureProfile
    return create_profile(config, pressure_factory, PressureProfile)


def create_chemistry(config):
    from taurex.data.profiles.chemistry.chemistry import Chemistry
    from taurex.data.profiles.chemistry.gas.gas import Gas

    gases = []
    config_key = []

    for key, value in config.items():

        if isinstance(value, dict):
            log.debug('FOUND GAS {} {}'.format(key, value))

            config_key.append(key)
            new_value = value
            new_value['molecule_name'] = key
            _gas = create_profile(new_value, gas_factory,
                                  baseclass=Gas, keyword_type='gas_type')
            gases.append(_gas)

    for k in config_key:
        del config[k]

    log.debug('leftover keys {}'.format(config))

    obj = create_profile(config, chemistry_factory, Chemistry,
                         keyword_type='chemistry_type')

    if hasattr(obj, 'addGas'):
        for g in gases:
            obj.addGas(g)

    return obj


def model_factory(model_type):
    cf = ClassFactory()
    for klass in cf.modelKlasses:
        try:
            if model_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('Model {} not implemented'.format(model_type))


def planet_factory(planet_type):
    cf = ClassFactory()
    for klass in cf.planetKlasses:
        try:
            if planet_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('Planet {} not implemented'.format(planet_type))


def optimizer_factory(optimizer):
    cf = ClassFactory()
    for klass in cf.optimizerKlasses:
        try:
            if optimizer in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('Optimizer {} not implemented'.format(optimizer))


def observation_factory(observation):
    cf = ClassFactory()
    for klass in cf.observationKlasses:
        try:
            if observation in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('Observation {} not '
                              'implemented'.format(observation))


def instrument_factory(instrument):
    cf = ClassFactory()
    for klass in cf.instrumentKlasses:
        try:
            if instrument in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)
        except AttributeError:
            pass

    raise NotImplementedError('Instrument {} not '
                              'implemented'.format(instrument))


def create_star(config):
    from taurex.data.stellar.star import Star
    config, klass, mixin = determine_klass(config, 'star_type', star_factory,
                                           Star)

    obj = klass(**config)

    return obj


def create_planet(config):
    from taurex.data.planet import Planet

    if 'planet_type' not in config:
        config['planet_type'] = 'simple'

    config, klass, mixin = \
        determine_klass(config, 'planet_type',
                        planet_factory, Planet)

    obj = klass(**config)

    return obj


def create_optimizer(config):
    from taurex.optimizer.optimizer import Optimizer
    config, klass, mixin = \
        determine_klass(config, 'optimizer',
                        optimizer_factory, Optimizer)

    obj = klass(**config)

    return obj


def create_observation(config):
    from taurex.spectrum import BaseSpectrum
    config, klass, mixin = \
        determine_klass(config, 'observation',
                        observation_factory, BaseSpectrum)

    obj = klass(**config)

    return obj


def determine_klass(config, field, factory, baseclass=None):
    from taurex.mixin.core import build_new_mixed_class
    try:
        klass_field = config.pop(field).lower()
    except KeyError:
        log.error('Field not defined in {}'.format(field))
        raise KeyError

    klass = None
    is_mixin = False
    if klass_field == 'custom':
        try:
            python_file = config.pop('python_file')
        except KeyError:
            log.error('No python file for custom profile/model')
            raise KeyError

        klass = detect_and_return_klass(python_file, baseclass)
    else:
        split = klass_field.split('+')
        if len(split) == 1:
            klass = factory(klass_field)
        else:
            is_mixin = True
            base_klass = factory(split[-1])

            def the_mixin_factory(x):
                return mixin_factory(x, baseclass)

            mixins = [the_mixin_factory(s) for s in split[:-1]]
            klass = build_new_mixed_class(base_klass, mixins)

    return config, klass, is_mixin


def create_instrument(config):
    from taurex.instruments.instrument import Instrument
    config, klass, is_mixin = \
        determine_klass(config, 'instrument',
                        instrument_factory, Instrument)

    obj = klass(**config)

    return obj


def generate_contributions(config):

    cf = ClassFactory()
    contributions = []
    check_key = [k for k, v in config.items() if isinstance(v, dict)]
    for key in config.keys():

        for klass in cf.contributionKlasses:
            try:
                if key in klass.input_keywords():
                    contributions.append(create_klass(config[key],
                                                      klass,
                                                      False))
                    check_key.pop(check_key.index(key))
                    break
            except NotImplementedError:
                log.warning('%s', klass)
            except AttributeError:
                pass

    if len(check_key) > 0:
        log.error(f'Unknown Contributions {check_key}')
        raise Exception(f'Unknown contributions {check_key}')

    return contributions


def create_prior(prior):
    from taurex.util.fitting import parse_priors

    prior_name, args = parse_priors(prior)
    cf = ClassFactory()
    for p in cf.priorKlasses:
        if prior_name in (p.__name__, p.__name__.lower(), p.__name__.upper(),):
            return p(**args)
    else:
        raise ValueError('Unknown Prior Type in input file')


def create_model(config, gas, temperature, pressure, planet, star, observation=None):
    from taurex.model import ForwardModel
    log.debug(config)
    config, klass, is_mixin = \
        determine_klass(config, 'model_type',
                        model_factory, ForwardModel)

    log.debug('Chosen_model is {}'.format(klass))
    kwargs = get_keywordarg_dict(klass, is_mixin)
    log.debug('Model kwargs {}'.format(kwargs))
    log.debug('---------------{} {}--------------'.format(gas,
                                                          gas.activeGases))
    if 'planet' in kwargs:
        kwargs['planet'] = planet
    if 'star' in kwargs:
        kwargs['star'] = star
    if 'chemistry' in kwargs:
        kwargs['chemistry'] = gas
    if 'temperature_profile' in kwargs:
        kwargs['temperature_profile'] = temperature
    if 'pressure_profile' in kwargs:
        kwargs['pressure_profile'] = pressure
    if 'observation' in kwargs:
        kwargs['observation'] = observation
    log.debug('New Model kwargs {}'.format(kwargs))
    log.debug('Creating model---------------')

    kwargs.update(dict([(k, v) for k, v in config.items()
                  if not isinstance(v, dict)]))
    obj = klass(**kwargs)

    contribs = generate_contributions(config)

    for c in contribs:
        obj.add_contribution(c)

    return obj


def detect_and_return_klass(python_file, baseclass):
    import importlib.util
    import inspect
    spec = importlib.util.spec_from_file_location("foo", python_file)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    classes = [m[1] for m in inspect.getmembers(foo, inspect.isclass) if m[1]
               is not baseclass and issubclass(m[1], baseclass)]

    if len(classes) == 0:
        log.error('Could not find class of type %s in file %s',
                  baseclass, python_file)
        raise Exception(f'No class inheriting from {baseclass} in '
                        f'{python_file}')
    return classes[0]
