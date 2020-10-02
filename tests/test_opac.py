import pytest
from taurex.opacity import InterpolatingOpacity
import numpy as np
from hypothesis import given,example,settings
from hypothesis.strategies import floats
class FakeOpac(InterpolatingOpacity):

    

    def __init__(self):
        super().__init__('Fake')

        num_temp = 5
        num_press = 8
        num_wav = 100

        self._temperature_grid = np.linspace(300, 1000, num_temp)
        self._pressure_grid = np.logspace(0, 6, num_press)
        self._wavenumber_grid = np.linspace(300, 3000, num_wav)
        self._xsec_grid = np.ones(num_temp*num_press*num_wav).reshape(num_press, num_temp, num_wav)

    @property
    def xsecGrid(self):
        return self._xsec_grid

    @property
    def wavenumberGrid(self):
        return self._wavenumber_grid

    @property
    def temperatureGrid(self):
        return self._temperature_grid

    @property
    def pressureGrid(self):
        return self._pressure_grid

@pytest.fixture(scope="module")
def fake_interp_opac():
    return FakeOpac()

@given(temperature=floats(100,2000), pressure=floats(1e-1, 1e7))
@example(temperature=100, pressure=1e-1)
@example(temperature=10, pressure=1e-1)
@example(temperature=2000, pressure=1e-1)
@example(temperature=3000, pressure=1e-1)
@example(temperature=100, pressure=1e-2)
@example(temperature=10, pressure=1e-2)
@example(temperature=2000, pressure=1e-2)
@example(temperature=3000, pressure=1e-2)
@example(temperature=100, pressure=1e7)
@example(temperature=10, pressure=1e7)
@example(temperature=2000, pressure=1e7)
@example(temperature=3000, pressure=1e7)
@example(temperature=100, pressure=1e8)
@example(temperature=10, pressure=1e8)
@example(temperature=2000, pressure=1e8)
@example(temperature=3000, pressure=1e8)
def test_find_closest_index(fake_interp_opac, temperature, pressure):
    import math
    t_min, t_max, p_min, p_max = \
        fake_interp_opac.find_closest_index(temperature, math.log10(pressure))

    t_grid = fake_interp_opac.temperatureGrid
    p_grid = fake_interp_opac.pressureGrid

    min_t = fake_interp_opac.temperatureMin
    max_t = fake_interp_opac.temperatureMax
    min_p = fake_interp_opac.pressureMin
    max_p = fake_interp_opac.pressureMax

    found_min_t = t_grid[t_min]
    found_max_t = t_grid[t_max]
    found_min_p = p_grid[p_min]
    found_max_p = p_grid[p_max]

    if temperature < min_t:
        assert found_min_t == min_t
    elif temperature > max_t:
        assert found_max_t == max_t
    else:
        assert found_min_t <= temperature
        assert found_max_t >= temperature

    if pressure < min_p:
        assert found_min_p == min_p
    elif pressure > max_p:
        assert found_max_p == max_p
    else:
        assert found_min_p <= pressure
        assert found_max_p >= pressure


def test_min_max(fake_interp_opac):
    assert fake_interp_opac.pressureMax == fake_interp_opac.pressureGrid.max()
    assert fake_interp_opac.pressureMin == fake_interp_opac.pressureGrid.min()
    assert fake_interp_opac.temperatureMin == fake_interp_opac.temperatureGrid.min()
    assert fake_interp_opac.temperatureMax == fake_interp_opac.temperatureGrid.max()


@given(temperature=floats(100, 2000), pressure=floats(1e-1, 1e7))
@example(temperature=100, pressure=1e-1)
@example(temperature=10, pressure=1e-1)
@example(temperature=2000, pressure=1e-1)
@example(temperature=3000, pressure=1e-1)
@example(temperature=100, pressure=1e-2)
@example(temperature=10, pressure=1e-2)
@example(temperature=2000, pressure=1e-2)
@example(temperature=3000, pressure=1e-2)
@example(temperature=100, pressure=1e7)
@example(temperature=10, pressure=1e7)
@example(temperature=2000, pressure=1e7)
@example(temperature=3000, pressure=1e7)
@example(temperature=100, pressure=1e8)
@example(temperature=10, pressure=1e8)
@example(temperature=2000, pressure=1e8)
@example(temperature=3000, pressure=1e8)
@settings(deadline=500)
def test_interpolation(fake_interp_opac, temperature, pressure):
    """ I cant test if its correct, only that it works for now"""
    op = fake_interp_opac.opacity(temperature, pressure)
    minimum_case = temperature < fake_interp_opac.temperatureMin \
        and pressure < fake_interp_opac.pressureMin

    maximum_case = pressure > fake_interp_opac.pressureMax \
        and temperature > fake_interp_opac.temperatureMax

    if minimum_case:
        assert np.allclose(
            np.zeros_like(fake_interp_opac.wavenumberGrid), op)

    elif maximum_case:
        assert np.array_equal(
            fake_interp_opac.xsecGrid[-1, -1]/10000, op)
    
