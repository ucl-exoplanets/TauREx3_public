from taurex.log import Logger

log = Logger('Factory')

def get_keywordarg_dict(klass):
    import inspect

    args, varargs, varkw, defaults = inspect.getargspec(klass.__init__)

    keyword_args = args[-len(defaults):]
    
    init_dicts = {}

    for keyword,value in zip(keyword_args,defaults):
        init_dicts[keyword] = value
    
    return init_dicts


def create_klass(config,klass):
    kwargs = get_keywordarg_dict(klass)

    for key in kwargs.keys():
        if key in config:
            value = config.pop(key)
            kwargs[key] = value
    
    obj = klass(**kwargs)

    for key,value in config.items():
        try:
            obj[key] = value
        except KeyError:
            log.warning('Object {} does not have parameter {}, skipping'.format(klass.__name__,key))
    return obj


def create_profile(config,factory):
    try:
        profile_type = config.pop('profile_type').lower()
    except KeyError:
        log.error('No profile_type defined input')
        raise KeyError    

    klass = factory(profile_type)

    obj = create_klass(config,klass)
    
    return obj

def gas_factory(profile_type):
    if profile_type == 'constant':
        from taurex.data.profiles.gas import ConstantGasProfile
        return ConstantGasProfile
    elif profile_type== 'twopoint':
        from taurex.data.profiles.gas import TwoPointGasProfile
        return TwoPointGasProfile
    elif profile_type == 'ace':
        from taurex.data.profiles.gas import ACEGasProfile
        return ACEGasProfile
    else:
        raise NotImplementedError('Gas profile {} not implemented'.format(profile_type))

def temp_factory(profile_type):
    if profile_type == 'isothermal':
        from taurex.data.profiles.temperature import Isothermal
        return Isothermal
    elif profile_type== 'guillot':
        from taurex.data.profiles.temperature import Guillot2010
        return Guillot2010
    else:
        raise NotImplementedError('Temperature profile {} not implemented'.format(profile_type))

def pressure_factory(profile_type):
    if profile_type == 'simple':
        from taurex.data.profiles.pressure import SimplePressureProfile
        return SimplePressureProfile
    else:
        raise NotImplementedError('Pressure profile {} not implemented'.format(profile_type))


def create_gas_profile(config):
    return create_profile(config,gas_factory)

def create_temperature_profile(config):
    return create_profile(config,temp_factory)

def create_pressure_profile(config):
    return create_profile(config,pressure_factory)

    
    
