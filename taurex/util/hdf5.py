
from .util import class_for_name
import h5py
import numpy as np

def get_klass_args(klass):
    import inspect
    init_dicts = {}
    args, varargs, varkw, defaults = inspect.getargspec(klass.__init__)

    keyword_args = args[-len(defaults):]

    return keyword_args

    return keyword_args

def load_generic_profile_from_hdf5(loc, label, module, identifier, profile_type=None):
    t_loc = loc[label]
    if profile_type is None:
        profile_type = t_loc[identifier][()]
    
    temp_keys = list(t_loc.keys())

    klass = class_for_name(module, profile_type)

    klass_kwargs = get_klass_args(klass)

    args_dict = {}

    for kw in klass_kwargs:
        
        if kw in temp_keys:
            v = t_loc[kw][()]
            if isinstance(v, np.ndarray) and v.dtype.type is np.string_:
                from taurex.util.util import decode_string_array
                v = decode_string_array(v)
            args_dict[kw] = v

    return klass(**args_dict)


def load_temperature_from_hdf5(loc):
    return load_generic_profile_from_hdf5(loc, 'Temperature',
                                          'taurex.data.profiles.temperature',
                                          'temperature_type')

def load_pressure_from_hdf5(loc):
    return load_generic_profile_from_hdf5(loc, 'Pressure',
                                          'taurex.data.profiles.pressure',
                                          'pressure_type')

def load_gas_from_hdf5(loc, molecule):
    return load_generic_profile_from_hdf5(loc, molecule,
                                          'taurex.data.profiles.chemistry',
                                          'gas_type')
def load_planet_from_hdf5(loc):
    return load_generic_profile_from_hdf5(loc, 'Planet',
                                          'taurex.data.planet','planet_type',
                                          'Planet')
def load_star_from_hdf5(loc):
    return load_generic_profile_from_hdf5(loc, 'Star',
                                          'taurex.data.stellar', 'star_type')

def load_chemistry_from_hdf5(loc):
    from taurex.data.profiles.chemistry import TaurexChemistry
    from taurex.util.util import decode_string_array
    chemistry = load_generic_profile_from_hdf5(loc, 'Chemistry',
                                               'taurex.data.profiles.chemistry',
                                               'chemistry_type')

    if isinstance(chemistry, TaurexChemistry):
        for mol in decode_string_array(loc['Chemistry']['active_gases'][()]):
            if mol not in chemistry._fill_gases:
                chemistry.addGas(load_gas_from_hdf5(loc['Chemistry'], mol))
        for mol in decode_string_array(loc['Chemistry']['inactive_gases']):
            if mol not in chemistry._fill_gases:
                chemistry.addGas(load_gas_from_hdf5(loc['Chemistry'], mol))

    return chemistry
