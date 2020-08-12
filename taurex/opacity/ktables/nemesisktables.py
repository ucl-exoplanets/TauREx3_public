from ..interpolateopacity import InterpolatingOpacity
import numpy as np
import pathlib
from .ktable import KTable
from taurex.util.util import sanitize_molecule_string


class NemesisKTables(KTable, InterpolatingOpacity):
    """
    This is the base class for computing opactities
    """

    @classmethod
    def discover(cls):
        import os
        import glob
        import pathlib
        from taurex.cache import GlobalCache
        
        path = GlobalCache()['ktable_path']
        if path is None:
            return []
        path = os.path.join(path, '*.kta')

        files = glob.glob(path)

        discovery = []

        interp = GlobalCache()['xsec_interpolation'] or 'linear'

        for f in files:
            splits = pathlib.Path(f).stem.split('_')
            mol_name = sanitize_molecule_string(splits[0])

            discovery.append((mol_name, [f, interp]))

        return discovery

    def __init__(self, filename, interpolation_mode='linear'):
        super().__init__('NemesisKtable:{}'.format(pathlib.Path(filename).stem[0:10]),
                        interpolation_mode=interpolation_mode)

        self._filename = filename
        splits = pathlib.Path(filename).stem.split('_')
        mol_name = sanitize_molecule_string(splits[0])
        self._molecule_name = mol_name
        self._spec_dict = None
        self._resolution = None
        self._decode_ktables(filename)
        
    @property
    def moleculeName(self):
        return self._molecule_name

    @property
    def xsecGrid(self):
        return self._xsec_grid

    def _decode_ktables(self, filename):
        self.debug('Reading NEMESIS FORMAT')
        nem_file_float = np.fromfile(filename, dtype=np.float32)
        nem_file_int = nem_file_float.view(np.int32)
        array_counter = 0
        self.debug('MAGIC NUMBER: %s', nem_file_int[0])
        wncount = nem_file_int[1]
        self.debug('WNCOUNT = %s', wncount)
        wnstart = nem_file_float[2]
        self.debug('WNSTART = %s um',wnstart)
        float_num = nem_file_float[3]
        self.debug('FLOAT: %s INT: %s',float_num, nem_file_int[4])

        num_pressure = nem_file_int[5]
        num_temperature = nem_file_int[6]
        num_quads = nem_file_int[7]

        self.debug('NP: %s NT: %s NQ: %s', num_pressure, num_temperature,
                   num_quads)
        self.debug('UNKNOWN VALUES: %s %s', nem_file_int[8], nem_file_int[9])

        array_counter += 10+num_quads*2
        self._samples, self._weights = \
            nem_file_float[10:array_counter].reshape(2, -1).astype(np.float64)
        self.debug('Samples: %s, Weights: %s', self._samples, self._weights)
        self.debug('%s', nem_file_int[array_counter])
        array_counter += 1
        self.debug('%s', nem_file_int[array_counter])
        array_counter += 1
        self._pressure_grid = nem_file_float[array_counter:array_counter+num_pressure].astype(np.float64)*1e5
        self.debug('Pgrid: %s',self._pressure_grid)
        array_counter+=num_pressure
        self._temperature_grid = nem_file_float[array_counter:array_counter+num_temperature].astype(np.float64)
        array_counter += num_temperature
        self.debug('Tgrid: %s',self._temperature_grid)
        self._wavenumber_grid = 10000/nem_file_float[array_counter:array_counter+wncount].astype(np.float64)
        self._wavenumber_grid = self._wavenumber_grid[::-1]
        array_counter += wncount
        self.debug('Wngrid: %s',self._wavenumber_grid)
    
        self._xsec_grid=(nem_file_float[array_counter:].reshape(wncount, num_pressure,num_temperature,num_quads) * 1e-20).astype(np.float64)
        self._xsec_grid = self._xsec_grid.transpose((1,2,0,3))
        self._xsec_grid = self._xsec_grid[::,::,::-1,:]
        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()
        # 0
        # 0
        # PRESSURE POINTS
        # TEMPERATUREPOINTS
        # WNGRID
        # KCOEFFS SCALED 1e-20    SHAPE(WNGRID,????)


    # def _load_pickle_file(self, filename):

    #     #Load the pickle file
    #     self.info('Loading opacity from {}'.format(filename))
    #     try:
    #         with open(filename, 'rb') as f:
    #             self._spec_dict = pickle.load(f)
    #     except UnicodeDecodeError:
    #         with open(filename, 'rb') as f:
    #             self._spec_dict = pickle.load(f, encoding='latin1')       
        
    #     self._wavenumber_grid = self._spec_dict['bin_centers']
    #     self._ngauss = self._spec_dict['ngauss']
    #     self._temperature_grid = self._spec_dict['t']
    #     self._pressure_grid = self._spec_dict['p']*1e5
    #     self._xsec_grid = self._spec_dict['kcoeff']
    #     self._weights = self._spec_dict['weights']
    #     self._molecule_name = self._spec_dict['name']

    #     self._min_pressure = self._pressure_grid.min()
    #     self._max_pressure = self._pressure_grid.max()
    #     self._min_temperature = self._temperature_grid.min()
    #     self._max_temperature = self._temperature_grid.max()
    #     self.clean_molecule_name()

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


        #return factor*(q_11*(Pmax-P)*(Tmax-T) + q_21*(P-Pmin)*(Tmax-T) + q_12*(Pmax-P)*(T-Tmin) + q_22*(P-Pmin)*(T-Tmin))
