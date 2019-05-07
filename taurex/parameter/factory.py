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



def create_gas_profile(config):
    try:
        profile_type = config.pop('profile_type').lower()
    except KeyError:
        log.error('No profile_type defined for gas profile in input')
        raise KeyError
    
    klass = None

    if profile_type == 'constant':
        from taurex.data.profiles.gas import ConstantGasProfile
        klass = ConstantGasProfile
    elif profile_type== 'twopoint':
        from taurex.data.profiles.gas import TwoPointGasProfile
        klass = TwoPointGasProfile
    elif profile_type == 'ace':
        from taurex.data.profiles.gas import ACEGasProfile
        klass = ACEGasProfile
    
    kwargs = get_keywordarg_dict(klass)

    for key in kwargs.keys():
        if key in config:
            value = config.pop(key)
            kwargs[key] = value
    
    gas = klass(**kwargs)

    for key,value in config.items():
        try:
            gas[key] = value
        except KeyError:
            log.warning('Profile {} does not have parameter {}, skipping'.format(profile_type,key))
    
    return gas



    
    
