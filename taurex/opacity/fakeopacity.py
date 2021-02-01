import numpy as np
from . import InterpolatingOpacity
from taurex.util.util import create_grid_res

class FakeOpacity(InterpolatingOpacity):

    def __init__(self, molecule_name, wn_res=15000, wn_size=(300,30000), num_p=20, num_t=27):
        super().__init__('FAKE')
        self._molecule_name = molecule_name
        self._wavenumber_grid = create_grid_res(wn_res, *wn_size)[:, 0]
        self._xsec_grid = np.random.rand(num_p, num_t, self._wavenumber_grid.shape[0])

        self._temperature_grid = np.linspace(100, 10000,num_t)
        self._pressure_grid = np.logspace(-6,6, num_p)
    @property
    def moleculeName(self):
        return self._molecule_name

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