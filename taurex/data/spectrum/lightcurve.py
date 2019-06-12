"""
Module dealing with observed lightcurves
"""

from .spectrum import BaseSpectrum
import numpy as np
from taurex.model.lightcurve.lightcurvedata import LightCurveData

class ObservedLightCurve(BaseSpectrum):
    """Loads an observed lightcurve from a pickle file.

    Parameters
    ----------

    filename : str
        Path to pickle file containing lightcurve data

    """

    def __init__(self,filename):
        super().__init__('observed_lightcurve')
        
        import pickle
        with open(filename,'rb') as f:
            lc_data = pickle.load(f,encoding='latin1')
            
        self.obs_spectrum = np.empty(shape=(len(lc_data['lc_info'][:, 0]), 4))
        self.obs_spectrum[:, 0] = lc_data['lc_info'][:, 0]
        self.obs_spectrum[:, 1] = lc_data['lc_info'][:, 3]
        self.obs_spectrum[:, 2] = lc_data['lc_info'][:, 1]
        self.obs_spectrum[:, 3] = lc_data['lc_info'][:, 2]

        self._spec,self._std = self._load_data_file(lc_data)



    def _load_data_file(self,lc_data):
        """load data from different instruments."""

        raw_data = []
        data_std = []

        for i in LightCurveData.availableInstruments:

            if i in lc_data['data']:
            # raw data includes data and datastd.
                raw_data.append(lc_data['data'][i][:len(lc_data['data'][i])//2])
                data_std.append(lc_data['data'][i][len(lc_data['data'][i])//2:])
        
        return np.concatenate(raw_data).flatten(),np.concatenate(data_std).flatten()


    @property
    def spectrum(self):
        """ 
        Returns Light curve spectrum.
        The lightcurve spectrum comes in the form of multiple lightcurves stuck together into
        one long spectrum. The number of lightcurves is equal to the number of bins in
        :func:`wavelengthGrid`.

        Returns
        -------
        spectrum : :obj:`array`

        """
        return self._spec

    @property
    def rawData(self):
        """
        Raw lightcurve data read from file

        Returns
        -------
        lc_data : :obj:`array`

        """
        self.obs_spectrum

    @property
    def wavelengthGrid(self):
        """
        Returns wavelength grid in microns

        Returns
        -------
        wlgrid : :obj:`array`

        """
        return self.obs_spectrum[:,0]
    


    @property
    def binEdges(self):
        """
        Returns bin edges for wavelength grid

        Returns
        -------
        out : :obj:`array`
        
        """
        return self.obs_spectrum[:, 3]
    
    @property
    def binWidths(self):
        """
        Widths for each bin in wavelength grid

        Returns
        -------
        out : :obj:`array`
        
        """
        return None


    @property
    def errorBar(self):
        """
        Like :func:`spectrum` except its the error at each point in the
        lightcurve spectrum

        Returns
        -------
        err : :obj:`array` 
            Error at each point in lightcurve spectrum
        
        """
        return self._std