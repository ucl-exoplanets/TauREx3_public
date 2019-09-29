
from .util import class_for_name
import h5py


def get_klass_args(klass):
    import inspect
    init_dicts = {}
    args, varargs, varkw, defaults = inspect.getargspec(klass.__init__)

    keyword_args = args[-len(defaults):]

    return keyword_args

    return keyword_args

def load_generic_profile_from_hdf5(loc, label, module, identifier):
    t_loc = loc[label]
    profile_type = t_loc[identifier][()]
    
    temp_keys = list(t_loc.keys())

    klass = class_for_name(module, profile_type)

    klass_kwargs = get_klass_args(klass)

    args_dict = {}

    for kw in klass_kwargs:
        
        if kw in temp_keys:
            args_dict[kw] = t_loc[kw][()]

    return klass(**args_dict)


def load_temperature_from_hdf5(loc):
    return load_generic_profile_from_hdf5(loc, 'Temperature',
                                          'taurex.data.profiles.temperature',
                                          'temperature_type')

def load_pressure_from_hdf5(loc):
    return load_generic_profile_from_hdf5(loc, 'Pressure',
                                          'taurex.data.profiles.pressure',
                                          'pressure_type')
