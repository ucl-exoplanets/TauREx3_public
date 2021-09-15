from ..interpolateopacity import InterpolatingOpacity
import pickle
import numpy as np
import pathlib
from .ktable import KTable


class PickleKTable(KTable, InterpolatingOpacity):
    """
    This is the base class for computing opactities
    """

    @classmethod
    def discover(cls):
        import os
        import glob
        import pathlib
        from taurex.cache import GlobalCache
        from taurex.util.util import sanitize_molecule_string

        path = GlobalCache()['ktable_path']
        if path is None:
            return []
        path = os.path.join(path, '*.pickle')

        files = glob.glob(path)

        discovery = []

        interp = GlobalCache()['xsec_interpolation'] or 'linear'

        for f in files:
            splits = pathlib.Path(f).stem.split('.')
            mol_name = sanitize_molecule_string(splits[0])

            discovery.append((mol_name, [f, interp]))

        return discovery

    def __init__(self, filename, interpolation_mode='linear'):
        super().__init__('PickleKtable:{}'.format(pathlib.Path(filename).stem[0:10]),
                        interpolation_mode=interpolation_mode)

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self._resolution = None
        self._load_pickle_file(filename)
        
        
    @property
    def moleculeName(self):
        return self._molecule_name

    @property
    def xsecGrid(self):
        return self._xsec_grid


    def _load_pickle_file(self, filename):

        #Load the pickle file
        self.info('Loading opacity from {}'.format(filename))
        try:
            with open(filename, 'rb') as f:
                self._spec_dict = pickle.load(f)
        except UnicodeDecodeError:
            with open(filename, 'rb') as f:
                self._spec_dict = pickle.load(f, encoding='latin1')       
        
        self._wavenumber_grid = self._spec_dict['bin_centers']
        self._ngauss = self._spec_dict['ngauss']
        self._temperature_grid = self._spec_dict['t']
        self._pressure_grid = self._spec_dict['p']*1e5
        self._xsec_grid = self._spec_dict['kcoeff']
        self._weights = self._spec_dict['weights']
        self._molecule_name = self._spec_dict['name']

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()
        self.clean_molecule_name()

    def clean_molecule_name(self):
        splits = self.moleculeName.split('_')
        self._molecule_name = splits[0]

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
    def weights(self):
        return self._weights