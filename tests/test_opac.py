import pytest
from taurex.opacity import InterpolatingOpacity
import numpy as np
from hypothesis import given,example
from hypothesis.strategies import floats
class FakeOpac(InterpolatingOpacity):

    def __init__(self):
        super().__init__('Fake')

        self._temperature_grid = np.linspace(300, 1000, 10)
        self._pressure_grid = np.logspace(0, 6, 11)
        self._wavenumber_grid = np.linspace(300, 3000, 100)
        self._xsec_grid = np.random.rand(10*11*100).reshape(11, 10, 100)

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

    t_min, t_max, p_min, p_max = fake_interp_opac.find_closest_index(temperature, pressure)

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
def test_interpolation(fake_interp_opac, temperature, pressure):
    """ I cant test if its correct, only that it works for now"""
    op = fake_interp_opac.opacity(temperature, pressure)

    minimum_case = temperature < fake_interp_opac.temperatureMin \
        and pressure < fake_interp_opac.pressureMin

    maximum_case = pressure > fake_interp_opac.pressureMax \
        and temperature > fake_interp_opac.temperatureMax

    if minimum_case:
        np.testing.assert_array_equal(
            np.zeros_like(fake_interp_opac.wavenumberGrid), op)

    if maximum_case:
        np.testing.assert_array_equal(
            fake_interp_opac.xsecGrid[-1, -1]/10000, op)
