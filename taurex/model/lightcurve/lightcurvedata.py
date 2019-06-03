from taurex.log import Logger
import logging
import numpy as np
class LightCurveData(Logger):


    @classmethod
    def fromInstrumentName(cls, name,lc_data):
        log = logging.getLogger()
        if name.lower() in ('wfc3',):
            return cls(lc_data,'wfc3',(1.1,1.8))
        elif name.lower() in ('spitzer',):
            return cls(lc_data,'spitzer',(3.4,8.2))
        elif name.lower() in ('stis',):
            return cls(lc_data,'stis',(0.3,1.0))
        else:
            log.error('LightCurve of instrument {} not recognized or implemented'.format(name))
            raise KeyError



    def __init__(self,lc_data,instrument_name,wavelength_region):
        super().__init__(self.__class__.__name__)
        self._instrument_name = instrument_name
        self._wavelength_region = wavelength_region
        self._load_data(lc_data)


    def _load_data(self,lc_data):
        self._time_series = lc_data['time_series'][self._instrument_name]
        self._raw_data = lc_data['data'][self._instrument_name][:len(lc_data['data'][self._instrument_name]) //2]
        self._data_std = lc_data['data'][self._instrument_name][len(lc_data['data'][self._instrument_name]) //2:]
        self._max_nfactor = np.max(self._raw_data, axis=1)
        self._min_nfactor = np.min(self._raw_data, axis=1)

        

    

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