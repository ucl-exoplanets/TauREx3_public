from taurex.log import Logger

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
    elif profile_type in ('guillot','guillot2010',):
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

def model_factory(model_type):
    if model_type =='transmission':
        from taurex.model import TransmissionModel
        return TransmissionModel
    elif model_type == 'emission':
        from taurex.model import EmissionModel
        return EmissionModel
    elif model_type in ('directimage', 'direct image'):
        from taurex.model import DirectImageModel
        return DirectImageModel
    else:
        raise NotImplementedError('Model {} not implemented'.format(model_type))

def optimizer_factory(optimizer):
    if optimizer == 'nestle':
        from taurex.optimizer.nestle import NestleOptimizer
        return NestleOptimizer
    elif optimizer in ('multinest','pymultinest',):
        from taurex.optimizer.multinest import MultiNestOptimizer
        return MultiNestOptimizer
    else:
        raise NotImplementedError('Optimizer {} not implemented'.format(optimizer))    


def create_optimizer(config):
    try:
        optimizer = config.pop('optimizer').lower()
    except KeyError:
        log.error('No optimizier defined input')
        raise KeyError    

    klass = optimizer_factory(optimizer)

    obj = klass(**config)
    
    return obj



def generate_contributions(config):
    from taurex.contributions import AbsorptionContribution,CIAContribution,RayleighContribution

    contributions = []
    for key in config.keys():
        if key == 'Absorption':
            contributions.append(create_klass(config[key],AbsorptionContribution))
        elif key == 'CIA':
            contributions.append(create_klass(config[key],CIAContribution))
        elif key == 'Rayleigh':
            contributions.append(create_klass(config[key],RayleighContribution))
        elif key == 'AbsorptionCUDA':
            from taurex.contributions.cuda.absorption import GPUAbsorptionContribution
            contributions.append(create_klass(config[key],GPUAbsorptionContribution))
        elif key == 'CIACUDA':
            from taurex.contributions.cuda.cia import GPUCIAContribution
            contributions.append(create_klass(config[key],GPUCIAContribution))
        elif key == 'RayleighCUDA':
            from taurex.contributions.cuda.rayleigh import GPURayleighContribution
            contributions.append(create_klass(config[key],GPURayleighContribution))
        elif key == 'SimpleClouds':
             from taurex.contributions import SimpleCloudsContribution
             contributions.append(create_klass(config[key],SimpleCloudsContribution))

    return contributions

    
def create_model(config,gas,temperature,pressure,planet,star):
    try:
        model_type = config.pop('model_type').lower()
    except KeyError:
        log.error('No model_type defined input')
        raise KeyError    

    klass = model_factory(model_type)
    log.debug('Chosen_model is {}'.format(klass))
    kwargs = get_keywordarg_dict(klass)
    log.debug('Model kwargs {}'.format(kwargs))
    log.debug('---------------{} {} {}--------------'.format(gas,gas.active_gases,gas.active_gas_mix_ratio))
    kwargs['planet'] = planet
    kwargs['star'] = star
    kwargs['gas_profile'] = gas
    kwargs['temperature_profile'] =temperature
    kwargs['pressure_profile'] = pressure
    log.debug('New Model kwargs {}'.format(kwargs))
    log.debug('Creating model---------------')
    obj = klass(**kwargs)
    
    contribs = generate_contributions(config)

    for c in contribs:
        obj.add_contribution(c)


    return obj       
