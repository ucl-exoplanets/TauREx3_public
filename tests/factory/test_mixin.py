

def test_mixin():

    from taurex.parameter.factory import determine_klass, temp_factory
    from taurex.temperature import TemperatureProfile, Isothermal
    from taurex.mixin.mixins import TempScaler
    from taurex.mixin.core import build_new_mixed_class
    config = {'profile_type': 'tempscalar+isothermal', 'T': 1500.0, 'scale_factor': 20.0}

    out_config, klass, is_mixin = determine_klass(config, 'profile_type', temp_factory, baseclass=TemperatureProfile)

    new_klass = build_new_mixed_class(Isothermal, TempScaler)

    assert is_mixin is True
    assert new_klass.__name__ == klass.__name__
    assert hasattr(klass, 'isoTemperature')
    assert hasattr(klass, 'scaleFactor')


def test_no_mixin():
    from taurex.parameter.factory import determine_klass, temp_factory
    from taurex.temperature import TemperatureProfile, Isothermal
    from taurex.mixin.mixins import TempScaler
    from taurex.mixin.core import build_new_mixed_class
    config = {'profile_type': 'isothermal', 'T': 1500.0}

    out_config, klass, is_mixin = determine_klass(config, 'profile_type', temp_factory, baseclass=TemperatureProfile)
    assert is_mixin is False
    assert hasattr(klass, 'isoTemperature')
    assert not hasattr(klass, 'scaleFactor')


def test_creation():
    from taurex.parameter.factory import create_temperature_profile, temp_factory
    from taurex.temperature import TemperatureProfile, Isothermal
    from taurex.mixin.mixins import TempScaler
    from taurex.mixin.core import build_new_mixed_class
    import numpy as np
    config = {'profile_type': 'tempscalar+isothermal', 'T': 1000.0, 'scale_factor': 20.0}

    temp = create_temperature_profile(config)

    assert temp.__class__ is not Isothermal
    assert temp.isoTemperature == 1000.0
    assert temp.scaleFactor == 20.0

    temp.initialize_profile(nlayers=100, pressure_profile=np.ones(100))

    assert np.all(temp.profile == 1000.0*20.0)
    params = temp.fitting_parameters()

    assert 'T_scale' in params
    assert 'T' in params
    assert params['T_scale'][2]() == 20
    assert params['T'][2]() == 1000.0

    params['T_scale'][3](5.0)
    params['T'][3](1000)

    temp.initialize_profile(nlayers=100, pressure_profile=np.ones(100))

    assert np.all(temp.profile == 1000.0*5.0)

