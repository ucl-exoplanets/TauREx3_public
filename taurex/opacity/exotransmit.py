import pathlib
from taurex.opacity.interpolateopacity import InterpolatingOpacity
import numpy as np


class ExoTransmitOpacity(InterpolatingOpacity):

    @classmethod
    def discover(cls):
        import os
        import glob
        import pathlib
        from taurex.cache import GlobalCache
        from taurex.util.util import sanitize_molecule_string

        path = GlobalCache()['xsec_path']
        if path is None:
            return []
        path = os.path.join(path, '*.dat')

        files = glob.glob(path)

        discovery = []

        interp = GlobalCache()['xsec_interpolation'] or 'linear'

        for f in files:
            mol_name = sanitize_molecule_string(pathlib.Path(f).stem[4:])
            discovery.append((mol_name, [f, interp]))

        return discovery

    def __init__(self, filename, interpolation_mode='linear'):

        super().__init__('ExoOpacity:{}'.format(
            pathlib.Path(filename).stem[4:]),
            interpolation_mode=interpolation_mode)

        self._filename = filename
        self._molecule_name = pathlib.Path(filename).stem[4:]
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

    BIBTEX_ENTRIES = ["""
        @ARTICLE{2017PASP..129d4402K,
            author = {{Kempton}, Eliza M. -R. and {Lupu}, Roxana and {Owusu-Asare}, Albert and {Slough}, Patrick and {Cale}, Bryson},
                title = "{Exo-Transmit: An Open-Source Code for Calculating Transmission Spectra for Exoplanet Atmospheres of Varied Composition}",
            journal = {Publications of the Astronomical Society of the Pacific},
            keywords = {Astrophysics - Earth and Planetary Astrophysics},
                year = 2017,
                month = apr,
            volume = {129},
            number = {974},
                pages = {044402},
                doi = {10.1088/1538-3873/aa61ef},
        archivePrefix = {arXiv},
            eprint = {1611.03871},
        primaryClass = {astro-ph.EP},
            adsurl = {https://ui.adsabs.harvard.edu/abs/2017PASP..129d4402K},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

        """,
    ]