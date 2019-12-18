import pathlib
from taurex.opacity.interpolateopacity import InterpolatingOpacity
import numpy as np


class ExoTransmitOpacity(InterpolatingOpacity):

    def __init__(self, filename, interpolation_mode='linear', in_memory=False):

        super().__init__('ExoOpacity:{}'.format(
                            pathlib.Path(filename).stem[4:]),
                         interpolation_mode=interpolation_mode)

        self._filename = filename
        self._molecule_name = pathlib.Path(filename).stem[4:]
        self.in_memory = in_memory
        self._load_exo_transmit(filename)

    def _load_exo_transmit(self, filename):
        self.debug('Loading opacity from {}'.format(filename))

        with open(filename, 'r') as f:
            lines = f.readlines()

        self._temperature_grid = np.array(
            [float(l) for l in lines[0].split()])  # *t_conversion

        self._pressure_grid = np.array(
            [float(l) for l in lines[1].split()])*1e5

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()

        wn_grid = []

        for ln in lines[2:]:
            arr = np.array([float(l) for l in ln.split()])
            if arr.shape[0] == 1:
                wn_grid.append(10000*1e-6/arr[0])

        wn_grid = np.array(wn_grid)

        grid_sort = wn_grid.argsort()
        self._wavenumber_grid = wn_grid[grid_sort]

        pressure_count = 0
        lambda_count = -1
        self._xsec_grid = np.empty(shape=(self.pressureGrid.shape[0],
                                          self.temperatureGrid.shape[0],
                                          self.wavenumberGrid.shape[0]))

        for ln in lines[2:]:
            arr = np.array([float(l) for l in ln.split()])
            if arr.shape[0] == 1:
                lambda_count += 1
                pressure_count = 0
            else:
                self._xsec_grid[pressure_count, :,
                                lambda_count] = arr[1:] + 1e-60
                pressure_count += 1

        self._xsec_grid = self._xsec_grid[:, :, grid_sort]*10000

    @property
    def wavenumberGrid(self):
        return self._wavenumber_grid

    @property
    def temperatureGrid(self):
        return self._temperature_grid

    @property
    def pressureGrid(self):
        return self._pressure_grid

    @property
    def resolution(self):
        return self._resolution

    @property
    def moleculeName(self):
        return self._molecule_name

    @property
    def xsecGrid(self):
        return self._xsec_grid
