import logging
from taurex.log import Logger
import numpy as np
from taurex.output.writeable import Writeable


class LightCurveData(Logger, Writeable):

    availableInstruments = ['wfc3', 'spitzer', 'stis', 'twinkle']

    @classmethod
    def fromInstrumentName(cls, name, lc_data):
        log = logging.getLogger()
        if name.lower() in ('wfc3',):
            return cls(lc_data, 'wfc3', (1.1, 1.8))
        elif name.lower() in ('spitzer',):
            return cls(lc_data, 'spitzer', (3.4, 8.2))
        elif name.lower() in ('stis',):
            return cls(lc_data, 'stis', (0.3, 1.0))
        elif name.lower() in ('twinkle',):
            return cls(lc_data, 'twinkle', (0.4, 4.5))
        else:
            log.error(
                'LightCurve of instrument %s not recognized'
                ' or implemented', name)
            raise KeyError

    def __init__(self, lc_data, instrument_name, wavelength_region):
        super().__init__(self.__class__.__name__)
        self._instrument_name = instrument_name
        # new version
        if self._instrument_name not in lc_data:
            self.error('Instrument with key %s not found in pickled lightcurve'
                       ' file', self._instrument_name)
            raise KeyError()

        self._wavelength_region = wavelength_region
        self._load_data(lc_data)

    def _load_data(self, lc_data):
        # new
        self._time_series = lc_data[self._instrument_name]['time_series']

        # new
        self._raw_data = lc_data[self._instrument_name]['data'][:, :, 0]
        self._data_std = lc_data[self._instrument_name]['data'][:, :, 1]

        self._max_nfactor = np.max(self._raw_data, axis=1)
        self._min_nfactor = np.min(self._raw_data, axis=1)

    @property
    def instrumentName(self):
        return self._instrument_name

    @property
    def wavelengthRegion(self):
        return self._wavelength_region

    @property
    def timeSeries(self):
        return self._time_series

    @property
    def rawData(self):
        return self._raw_data

    @property
    def dataError(self):
        return self._data_std

    @property
    def minNFactors(self):
        return self._min_nfactor

    @property
    def maxNFactors(self):
        return self._max_nfactor

    def write(self, output):

        lc_grp = output.create_group(self.instrumentName)

        lc_grp.write_array('raw_data', self.rawData)
        lc_grp.write_array('data_error', self.dataError)
        lc_grp.write_array('min_n_factors', self.minNFactors)
        lc_grp.write_array('max_n_factors', self.maxNFactors)
        lc_grp.write_array('time_series', self.timeSeries)
        lc_grp.write_scalar('min_wavelength', self.wavelengthRegion[0])
        lc_grp.write_scalar('max_wavelength', self.wavelengthRegion[1])

        return lc_grp
