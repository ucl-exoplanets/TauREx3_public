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





def create_profile(config, factory, baseclass=None):
    config, klass = determine_klass(config, 'profile_type',factory,
                                    baseclass)

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
    elif profile_type in ('file',):
        from taurex.data.profiles.temperature import TemperatureFile
        return TemperatureFile
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
    from taurex.data.profiles.chemistry.gas import Gas
    return create_profile(config, gas_factory, Gas)


def create_temperature_profile(config):
    from taurex.data.profiles.temperature import TemperatureProfile
    return create_profile(config, temp_factory, TemperatureProfile)


def create_pressure_profile(config):
    from taurex.data.profiles.pressure.pressureprofile import PressureProfile
    return create_profile(config, pressure_factory, PressureProfile)


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
    elif chemistry in ('file', ):
        from taurex.chemistry import ChemistryFile
        return create_klass(config, ChemistryFile)
    elif chemistry in ('custom',):
        from taurex.chemistry import Chemistry
        config['chemistry_type'] = 'custom'
        config, klass = determine_klass(config, 'chemistry_type', None,
                        Chemistry)
        obj = klass(**config)
        return obj
    elif chemistry in ('taurex', 'complex', 'defined', 'free'):
        from taurex.data.profiles.chemistry import TaurexChemistry
        gases = []
        config_key=[]
        for key, value in config.items():
            
            if isinstance(value, dict):
                log.debug('FOUND GAS {} {}'.format(key, value))
                config_key.append(key)
                gas_type = value.pop('gas_type').lower()
                klass = gas_factory(gas_type)
                gases.append(klass(molecule_name=key, **value))

        for k in config_key: del config[k]
        log.debug('leftover keys {}'.format(config))
        obj = TaurexChemistry(**config)
        for g in gases:
            obj.addGas(g)
    
        return obj
    else:
        raise ValueError('Unknown chemistry type {}'.format(chemistry))


def model_factory(model_type):
    if model_type == 'transmission':
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


def planet_factory(planet_type):
    if planet_type in ('simple', 'planet', 'basic', 'meme',):
        from taurex.data import Planet
        return Planet
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


def instrument_factory(instrument):
    if instrument in ('file', 'fromfile',):
        from taurex.instruments import InstrumentFile
        return InstrumentFile


def create_star(config):
    from taurex.data.stellar.star import BlackbodyStar
    config, klass = determine_klass(config, 'star_type', star_factory,
                                    BlackbodyStar)

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
    from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution

    contributions = []
    for key in config.keys():
        if key == 'Absorption':
            contributions.append(create_klass(config[key],AbsorptionContribution))
        elif key == 'CIA':
            contributions.append(create_klass(config[key],CIAContribution))
        elif key == 'Rayleigh':
            contributions.append(create_klass(config[key],RayleighContribution))
        elif key == 'SimpleClouds':
             from taurex.contributions import SimpleCloudsContribution
             contributions.append(create_klass(config[key],SimpleCloudsContribution))
        elif key == 'BHMie':
             from taurex.contributions import BHMieContribution
             contributions.append(create_klass(config[key],BHMieContribution))
        elif key == 'LeeMie':
             from taurex.contributions import LeeMieContribution
             contributions.append(create_klass(config[key],LeeMieContribution))
        elif key == 'FlatMie':
             from taurex.contributions import FlatMieContribution
             contributions.append(create_klass(config[key],FlatMieContribution))

    return contributions

    
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
        self.error('Could not find class of type %s in file %s',baseclass, python_file)
        raise Exception(f'No class inheriting from {baseclass} in '
                        f'{python_file}')
    return classes[0]
