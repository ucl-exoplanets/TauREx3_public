
from .util import class_for_name
import numpy as np


def get_klass_args(klass):
    import inspect
    args, varargs, varkw, defaults = inspect.getargspec(klass.__init__)
    if defaults is None:
        return []
    keyword_args = args[-len(defaults):]

    return keyword_args


def load_generic_profile_from_hdf5(loc, module, identifier, 
                                   profile_type=None, premade_dict=None,
                                   replacement_dict=None):

    if profile_type is None:
        profile_type = loc[identifier][()]

    temp_keys = list(loc.keys())
    klass = class_for_name(module, profile_type)

    klass_kwargs = get_klass_args(klass)

    args_dict = premade_dict or {}
    repl_dict = replacement_dict or {}

    for kw in klass_kwargs:

        if kw in temp_keys:
            v = loc[kw][()]
            if isinstance(v, np.ndarray) and v.dtype.type is np.string_:
                from taurex.util.util import decode_string_array
                v = decode_string_array(v)
            if kw in repl_dict:
                args_dict[kw] = repl_dict[kw]
            else:
                args_dict[kw] = v
            

    return klass(**args_dict)


def load_temperature_from_hdf5(loc,replacement_dict=None):
    return load_generic_profile_from_hdf5(loc['Temperature'],
                                          'taurex.data.profiles.temperature',
                                          'temperature_type',
                                          replacement_dict=replacement_dict)


def load_pressure_from_hdf5(loc,replacement_dict=None):
    return load_generic_profile_from_hdf5(loc['Pressure'],
                                          'taurex.data.profiles.pressure',
                                          'pressure_type',
                                          replacement_dict=replacement_dict)


def load_gas_from_hdf5(loc, molecule, replacement_dict=None):
    return load_generic_profile_from_hdf5(loc[molecule],
                                          'taurex.data.profiles.chemistry',
                                          'gas_type',
                                          replacement_dict=replacement_dict)


def load_planet_from_hdf5(loc,replacement_dict=None):
    return load_generic_profile_from_hdf5(loc['Planet'],
                                          'taurex.data.planet','planet_type',
                                          profile_type='Planet',
                                          replacement_dict=replacement_dict)


def load_star_from_hdf5(loc,replacement_dict=None):
    return load_generic_profile_from_hdf5(loc['Star'],
                                          'taurex.data.stellar', 'star_type',
                                          replacement_dict=replacement_dict)


def load_chemistry_from_hdf5(loc,replacement_dict=None):
    from taurex.data.profiles.chemistry import TaurexChemistry
    from taurex.util.util import decode_string_array
    chemistry = load_generic_profile_from_hdf5(loc['Chemistry'],
                                               'taurex.data.profiles.chemistry',
                                               'chemistry_type',
                                          replacement_dict=replacement_dict)

    if isinstance(chemistry, TaurexChemistry):
        for mol in decode_string_array(loc['Chemistry']['active_gases'][()]):
            if mol not in chemistry._fill_gases:
                chemistry.addGas(load_gas_from_hdf5(loc['Chemistry'], mol))
        for mol in decode_string_array(loc['Chemistry']['inactive_gases']):
            if mol not in chemistry._fill_gases:
                chemistry.addGas(load_gas_from_hdf5(loc['Chemistry'], mol))

    return chemistry


def load_contrib_from_hdf5(loc, contribution,replacement_dict=None):
    return load_generic_profile_from_hdf5(loc[contribution],
                                          'taurex.contributions',
                                          'contrib_type',
                                          profile_type=contribution,
                                          replacement_dict=replacement_dict)




def load_model_from_hdf5(loc, replacement_dict=None):
    chemistry = load_chemistry_from_hdf5(loc, replacement_dict=replacement_dict)
    pressure = load_pressure_from_hdf5(loc, replacement_dict=replacement_dict)
    temperature = load_temperature_from_hdf5(loc, replacement_dict=replacement_dict)
    planet = load_planet_from_hdf5(loc, replacement_dict=replacement_dict)
    star = load_star_from_hdf5(loc, replacement_dict=replacement_dict)

    kwargs ={}

    kwargs['planet'] = planet
    kwargs['star'] = star
    kwargs['chemistry'] = chemistry
    kwargs['temperature_profile'] = temperature
    kwargs['pressure_profile'] = pressure

    model = load_generic_profile_from_hdf5(loc, 'taurex.model',  'model_type',
                                           premade_dict=kwargs,
                                           replacement_dict=replacement_dict)

    contribution_loc = loc['Contributions']

    def contrib_iterator(loc):
        import h5py
        for key in loc.keys():
            item = loc[key]
            if isinstance(item, h5py.Group):
                yield key

    for contrib in contrib_iterator(contribution_loc):
        model.add_contribution(load_contrib_from_hdf5(contribution_loc,
                                                      contrib,
                                                      replacement_dict=replacement_dict))

    return model


def taurex_hdf5_to_model(filename, replacement_dict=None):

    import h5py
    with h5py.File(filename, 'r') as f:
        model = load_model_from_hdf5(f['ModelParameters'],
                                     replacement_dict=replacement_dict)

    return model


def taurex_hdf5_to_observation(filename):

    import h5py
    from taurex.util.util import wnwidth_to_wlwidth
    from taurex.data.spectrum import ArraySpectrum
    with h5py.File(filename, 'r') as f:
        try:
            instrument_section = f['Output']['Spectra']
        except KeyError:
            raise KeyError('No instrument data found in HDF5 or retrieval hdf5 used')
        inst_wngrid = instrument_section['instrument_wngrid'][...]
        inst_spectrum = instrument_section['instrument_spectrum'][...]
        inst_noise = instrument_section['instrument_noise'][...]
        inst_width = instrument_section['instrument_wnwidth'][...]

        inst_wlgrid = 10000/inst_wngrid

        inst_wlwidth = wnwidth_to_wlwidth(inst_wngrid, inst_width)
        observation = ArraySpectrum(np.vstack([inst_wlgrid, inst_spectrum,
                                               inst_noise, inst_wlwidth]).T)
    return observation
