import pytest
from taurex.pressure import SimplePressureProfile
from hypothesis import given, assume
from hypothesis.strategies import floats, integers
import numpy as np


@given(min_pressure=floats(1e4, 1e6), max_pressure=floats(1e-5,1e5), 
       nlayers=integers(-100, 400))
def test_simple_pressure(min_pressure, max_pressure, nlayers):

    if min_pressure > max_pressure or nlayers <= 0:
        with pytest.raises(ValueError):
            sp = SimplePressureProfile(nlayers=nlayers, 
                                       atm_min_pressure=min_pressure,
                                       atm_max_pressure=max_pressure)
    else:
        sp = SimplePressureProfile(nlayers=nlayers, 
                                    atm_min_pressure=min_pressure,
                                    atm_max_pressure=max_pressure)

        assert sp.profile is None

        sp.compute_pressure_profile()

        pressure_profile = sp.profile

        pressure_profile_levels = sp.pressure_profile_levels


        assert pressure_profile_levels.argmax() == 0 # Ensure maximum is the first
        assert pressure_profile_levels.argmin() == nlayers-1 # ensure minimum is last
        assert pressure_profile_levels.max() == max_pressure
        assert pressure_profile_levels.min() == min_pressure

        # Test to ensure they are bounded correctly
        assert np.all(pressure_profile_levels[1:-1] < max_pressure)
        assert np.all(pressure_profile_levels[1:-1] > min_pressure)

        # Test to ensure it is descreasing c
        assert np.all(np.diff(pressure_profile_levels) < 0)

        # Ensure pressure profile is always between levels

        assert np.all(pressure_profile_levels[:-1] > pressure_profile)
        assert np.all(pressure_profile_levels[1:] < pressure_profile)
        
