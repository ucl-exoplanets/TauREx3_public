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
        from taurex.data.profiles.chemistry import ConstantGas
        return ConstantGas
    elif profile_type in ('twopoint', '2point',):
        from taurex.data.profiles.chemistry import TwoPointGas
        return TwoPointGas
    elif profile_type in ('twolayer','2layer',):
        from taurex.data.profiles.chemistry import TwoLayerGas
        return TwoLayerGas
    else:
        raise NotImplementedError('Gas profile {} not implemented'.format(profile_type))

def temp_factory(profile_type):
    if profile_type == 'isothermal':
        from taurex.data.profiles.temperature import Isothermal
        return Isothermal
    elif profile_type in ('guillot','guillot2010',):
        from taurex.data.profiles.temperature import Guillot2010
        return Guillot2010
    elif profile_type in ('npoint',):
        from taurex.data.profiles.temperature import NPoint
        return NPoint
    elif profile_type in ('rodgers','rodgers2010',):
        from taurex.data.profiles.temperature import Rodgers2000
        return Rodgers2000
    else:
        raise NotImplementedError('Temperature profile {} not implemented'.format(profile_type))

def pressure_factory(profile_type):
    if profile_type == 'simple':
        from taurex.data.profiles.pressure import SimplePressureProfile
        return SimplePressureProfile
    else:
        raise NotImplementedError('Pressure profile {} not implemented'.format(profile_type))

def star_factory(star_type):
    if star_type == 'blackbody':
        from taurex.data.stellar import BlackbodyStar
        return BlackbodyStar
    elif star_type == 'phoenix':
        from taurex.data.stellar import PhoenixStar
        return PhoenixStar
    else:
        raise NotImplementedError('Star of type {} not implemented'.format(star_type))


def create_gas_profile(config):
    return create_profile(config,gas_factory)

def create_temperature_profile(config):
    return create_profile(config,temp_factory)

def create_pressure_profile(config):
    return create_profile(config,pressure_factory)

def create_ace(config):
    from taurex.data.profiles.chemistry import ACEChemistry
    return ACEChemistry(**config)

def create_chemistry(config):
    try:
        chemistry = config.pop('chemistry_type').lower()
    except KeyError:
        log.error('No chemistry defined in input')
        raise KeyError    

    if chemistry in ('ace','equilibrium'):
        return create_ace(config)
    elif chemistry in ('taurex','complex','custom','defined'):
        from taurex.data.profiles.chemistry import TaurexChemistry
        gases = []
        config_key=[]
        for key,value in config.items():
            
            if isinstance(value,dict):
                log.debug('FOUND GAS {} {}'.format(key,value))
                config_key.append(key)
                gas_type = value.pop('gas_type').lower()
                klass = gas_factory(gas_type)
                gases.append(klass(molecule_name=key,**value))

        for k in config_key: del config[k]
        log.debug('leftover keys {}'.format(config))
        obj = TaurexChemistry(**config)
        for g in gases:
            obj.addGas(g)
    
        return obj
    else:
        raise ValueError('Unknown chemistry type {}'.format(chemistry))



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
    elif optimizer in ('polychord','pypolychord'):
        from taurex.optimizer.polychord import PolyChordOptimizer
        return PolyChordOptimizer
    elif optimizer in ('dypolychord','dynamic-polychord'):
        from taurex.optimizer.dypolychord import dyPolyChordOptimizer
        return dyPolyChordOptimizer
    else:
        raise NotImplementedError('Optimizer {} not implemented'.format(optimizer))    


def create_star(config):
    try:
        star = config.pop('star_type').lower()
    except KeyError:
        log.error('No star defined input')
        raise KeyError    

    klass = star_factory(star)

    obj = klass(**config)
    
    return obj

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
        elif key == 'Mie':
             from taurex.contributions import MieContribution
             contributions.append(create_klass(config[key],MieContribution))

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
    log.debug('---------------{} {}--------------'.format(gas,gas.activeGases))
    kwargs['planet'] = planet
    kwargs['star'] = star
    kwargs['chemistry'] = gas
    kwargs['temperature_profile'] =temperature
    kwargs['pressure_profile'] = pressure
    log.debug('New Model kwargs {}'.format(kwargs))
    log.debug('Creating model---------------')
    obj = klass(**kwargs)
    
    contribs = generate_contributions(config)

    for c in contribs:
        obj.add_contribution(c)


    return obj       
