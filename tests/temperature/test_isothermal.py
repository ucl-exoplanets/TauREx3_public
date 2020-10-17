import pytest
from hypothesis import given
from hypothesis.strategies import floats, integers
from taurex.temperature import Isothermal
import numpy as np

@given(temperature=floats(allow_nan=False),
       another_temperature=floats(allow_nan=False),
       nlayers=integers(1, 50))
def test_isothermal(temperature, another_temperature, nlayers):

    iso = Isothermal(T=temperature)

    iso.initialize_profile(nlayers=nlayers)

    params = iso.fitting_parameters()

    assert iso.isoTemperature == temperature
    assert np.all(iso.profile == temperature)
    assert params['T'][2]() == temperature
    assert iso.profile.shape[0] == nlayers

    params['T'][3](another_temperature)

    assert iso.isoTemperature == another_temperature
    assert np.all(iso.profile == another_temperature)
    assert params['T'][2]() == another_temperature

