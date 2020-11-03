from taurex.log import Logger
from .classfactory import ClassFactory
log = Logger('Factory')

def get_keywordarg_dict(klass):
    import inspect
    init_dicts = {}
    args, varargs, varkw, defaults = inspect.getargspec(klass.__init__)
    log.debug('Inpection {} {} {} {}'.format(args, varargs, varkw, defaults))
    if defaults is None:
        return init_dicts

    keyword_args = args[-len(defaults):]
    
    

    for keyword,value in zip(keyword_args,defaults):
        init_dicts[keyword] = value
    
    return init_dicts



def create_klass(config, klass):
    kwargs = get_keywordarg_dict(klass)

    for key in config:
        if key in kwargs:
            value = config[key]
            kwargs[key] = value
        else:
            log.error('Object {} does not have parameter {}'.format(klass.__name__,key))
            log.error('Available parameters are %s',kwargs.keys())
            raise KeyError
    obj = klass(**kwargs)
    # for key,value in config.items():
    #     try:
    #         obj[key] = value
    #     except KeyError:

    #         raise KeyError
    return obj


def create_profile(config, factory, baseclass=None,
                   keyword_type='profile_type'):
    config, klass = determine_klass(config, keyword_type, 
                                    factory, baseclass)

    obj = create_klass(config, klass)

    return obj


def gas_factory(profile_type):
    cf = ClassFactory()

    for klass in cf.gasKlasses:
        try:
            if profile_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s',klass)

    raise NotImplementedError('Gas profile {} not implemented'.format(profile_type))


def temp_factory(profile_type):
    cf = ClassFactory()

    for klass in cf.temperatureKlasses:
        try:
            if profile_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s',klass)

    raise NotImplementedError('Temperature profile {} not implemented'.format(profile_type))


def chemistry_factory(profile_type):
    cf = ClassFactory()
    for klass in cf.chemistryKlasses:
        try:
            if profile_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s',klass)

    raise NotImplementedError('Chemistry {} not implemented'.format(profile_type))


def pressure_factory(profile_type):
    cf = ClassFactory()
    for klass in cf.pressureKlasses:
        try:
            if profile_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s',klass)

    raise NotImplementedError('Pressure profile {} not implemented'.format(profile_type))


def star_factory(star_type):
    cf = ClassFactory()
    for klass in cf.starKlasses:
        try:
            if star_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s',klass)

    raise NotImplementedError('Star of type {} not implemented'.format(star_type))


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
    from taurex.data.profiles.chemistry import TaurexChemistry
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

    if isinstance(obj, TaurexChemistry):
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
            log.warning('%s',klass)

    raise NotImplementedError('Model {} not implemented'.format(model_type))


def planet_factory(planet_type):
    cf = ClassFactory()
    for klass in cf.planetKlasses:
        try:
            if planet_type in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)

            
    raise NotImplementedError('Planet {} not implemented'.format(planet_type))


def optimizer_factory(optimizer):
    cf = ClassFactory()
    for klass in cf.optimizerKlasses:
        try:
            if optimizer in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s',klass)

    raise NotImplementedError('Optimizer {} not implemented'.format(optimizer))

def observation_factory(observation):
    cf = ClassFactory()
    for klass in cf.observationKlasses:
        try:
            if observation in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s', klass)

    raise NotImplementedError('Observation {} not implemented'.format(observation))



def instrument_factory(instrument):
    cf = ClassFactory()
    for klass in cf.instrumentKlasses:
        try:
            if instrument in klass.input_keywords():
                return klass
        except NotImplementedError:
            log.warning('%s',klass)

    raise NotImplementedError('Instrument {} not implemented'.format(instrument))


def create_star(config):
    from taurex.data.stellar.star import Star
    config, klass = determine_klass(config, 'star_type', star_factory,
                                    Star)

    obj = klass(**config)

    return obj


def create_planet(config):
    from taurex.data.planet import Planet

    if 'planet_type' not in config:
        config['planet_type'] = 'simple'

    config, klass = determine_klass(config, 'planet_type', planet_factory,
                                    Planet)

    obj = klass(**config)

    return obj


def create_optimizer(config):
    from taurex.optimizer.optimizer import Optimizer
    config, klass = determine_klass(config, 'optimizer', optimizer_factory,
                                    Optimizer)

    obj = klass(**config)
    
    return obj


def create_observation(config):
    from taurex.spectrum import BaseSpectrum
    config, klass = determine_klass(config, 'observation', observation_factory,
                                    BaseSpectrum)

    obj = klass(**config)
    
    return obj


def determine_klass(config, field, factory, baseclass=None):

    try:
        klass_field = config.pop(field).lower()
    except KeyError:
        log.error('Field not defined in {}'.format(field))
        raise KeyError

    klass = None
    if klass_field == 'custom':
        try:
            python_file = config.pop('python_file').lower()
        except KeyError:
            log.error('No python file for custom profile/model')
            raise KeyError

        klass = detect_and_return_klass(python_file, baseclass)
    else:
        klass = factory(klass_field)

    return config, klass


def create_instrument(config):
    from taurex.instruments.instrument import Instrument
    config, klass = determine_klass(config, 'instrument', instrument_factory,
                                    Instrument)

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
                    contributions.append(create_klass(config[key], klass))
                    check_key.pop(check_key.index(key))
                    break
            except NotImplementedError:
                log.warning('%s',klass)

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


def create_model(config,gas,temperature,pressure,planet,star):
    from taurex.model import ForwardModel
    log.debug(config)
    config, klass = determine_klass(config, 'model_type', model_factory,
                                    ForwardModel)

    log.debug('Chosen_model is {}'.format(klass))
    kwargs = get_keywordarg_dict(klass)
    log.debug('Model kwargs {}'.format(kwargs))
    log.debug('---------------{} {}--------------'.format(gas,gas.activeGases))
    if 'planet' in kwargs:
        kwargs['planet'] = planet
    if 'star' in kwargs:
        kwargs['star'] = star
    if 'chemistry' in kwargs:
        kwargs['chemistry'] = gas
    if 'temperature_profile' in kwargs:
        kwargs['temperature_profile'] =temperature
    if 'pressure_profile' in kwargs:
        kwargs['pressure_profile'] = pressure
    log.debug('New Model kwargs {}'.format(kwargs))
    log.debug('Creating model---------------')
    
    kwargs.update(dict([(k,v) for k,v in config.items() if not isinstance(v,dict)]))
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
    classes = [m[1] for m in inspect.getmembers(foo, inspect.isclass) if m[1] \
               is not baseclass and issubclass(m[1],baseclass)]

    if len(classes) == 0:
        log.error('Could not find class of type %s in file %s',baseclass, python_file)
        raise Exception(f'No class inheriting from {baseclass} in '
                        f'{python_file}')
    return classes[0]
