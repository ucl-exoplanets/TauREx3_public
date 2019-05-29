from .spectrum import BaseSpectrum
import numpy as np


class ObservedLightCurve(BaseSpectrum):
    """Loads an observed lightcurve from a file and computes bin
    edges and bin widths

    Parameters
    -----------
    filename: string
        File name of observed spectrum. Spectrum must be 3-4 columns
        1st column: 
            wavelength
        2nd column:
            spectral data
        3rd column:
            error
        4th(optional):
            bin width

    """

    def __init__(self,filename):
        super().__init__('observed_lightcurve')
        
        import pickle
        with open(filename,'rb') as f:
            lc_data = pickle.load(f,encoding='latin1')
            wfc_data = lc_data['data']['wfc3']
        self.obs_spectrum = np.empty(shape=(len(lc_data['lc_info'][:, 0]), 4))
        self.obs_spectrum[:, 0] = lc_data['lc_info'][:, 0]
        self.obs_spectrum[:, 1] = lc_data['lc_info'][:, 3]
        self.obs_spectrum[:, 2] = lc_data['lc_info'][:, 1]
        self.obs_spectrum[:, 3] = lc_data['lc_info'][:, 2]
        
        total_wfc = wfc_data.shape[0]//2

        self._spec = wfc_data[:total_wfc].flatten()
        self._std = wfc_data[total_wfc:].flatten()

    @property
    def spectrum(self):
        return self._spec

    @property
    def rawData(self):
        self.obs_spectrum

    @property
    def wavelengthGrid(self):
        return self.obs_spectrum[:,0]
    

    @property
    def binEdges(self):
        return self.obs_spectrum[:, 3]
    
    @property
    def binWidths(self):
        return None


    @property
    def errorBar(self):
        return self._std