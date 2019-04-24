from .spectrum import BaseSpectrum
import numpy as np

class ObservedSpectrum(BaseSpectrum):

    def __init__(self,filename):
        self._filename = filename

        self.info('Reading observed spectrum from file: {}'.format(self._filename))

        self._obs_spectrum = None
        self._bin_widths = None
        self._bin_edges = None

        self._read_file()
        self._process_spectrum()



    def _read_file(self):
        self._obs_spectrum = np.loadtxt(self.filename)
        self._obs_spectrum = self._obs_spectrum[self._obs_spectrum[:,0].argsort(axis=0)[::-1]]

    def _process_spectrum(self):
        if self.spectrum.shape[1] == 4:
            self._bin_widths = self._obs_spectrum[:,3]
            obs_wl = self.wavelengthGrid[::-1]
            obs_bw = self.binWidths[::-1]

            bin_edges = np.zeros(shape=(len(self.binWidths)*2,))

            bin_edges[0::2] = (obs_wl - obs_bw)/2
            bin_edges[1::2] = (obs_wl + obs_bw)/2

            self._bin_edges = bin_edges[::-1]



    @property
    def spectrum(self):
        return self._obs_spectrum

    @property
    def wavelengthGrid(self):
        return self.spectrum[:,0]

    
    @property
    def wavenumberGrid(self):
        return 10000/self.wavelengthGrid

    @property
    def binEdges(self):
        return self._bin_edges
    @property
    def binWidths(self):
        return self._bin_widths



    def manual_binning(self):


        
        