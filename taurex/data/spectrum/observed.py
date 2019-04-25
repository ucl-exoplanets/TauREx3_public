from .spectrum import BaseSpectrum
import numpy as np

class ObservedSpectrum(BaseSpectrum):
    """Loads an observed spectrum from a file and computes bin
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
        super().__init__('observed_spectrum')
        self._filename = filename

        self.info('Reading observed spectrum from file: {}'.format(self._filename))

        self._obs_spectrum = None
        self._bin_widths = None
        self._bin_edges = None


        self._read_file()
        self._process_spectrum()



    def _read_file(self):
        self._obs_spectrum = np.loadtxt(self._filename)
        self._obs_spectrum = self._obs_spectrum[self._obs_spectrum[:,0].argsort(axis=0)[::-1]]

    def _process_spectrum(self):
        if self.rawData.shape[1] == 4:
            self._bin_widths = self._obs_spectrum[:,3]
            obs_wl = self.wavelengthGrid[::-1]
            obs_bw = self.binWidths[::-1]

            bin_edges = np.zeros(shape=(len(self.binWidths)*2,))

            bin_edges[0::2] = obs_wl - obs_bw/2
            bin_edges[1::2] = obs_wl + obs_bw/2
            #bin_edges[-1] = obs_wl[-1]-obs_bw[-1]/2.

            self._bin_edges = bin_edges[::-1]
        else:
            self.manual_binning()


    @property
    def rawData(self):
        """Data read from file"""
        return self._obs_spectrum

    @property
    def spectrum(self):
        """The spectrum itself"""
        return self._obs_spectrum[:,1]


    @property
    def wavelengthGrid(self):
        """Wavelength grid in microns"""
        return self.rawData[:,0]

    
    @property
    def wavenumberGrid(self):
        """Wavenumber grid in cm-1"""
        return 10000/self.wavelengthGrid

    @property
    def binEdges(self):
        return self._bin_edges
    @property
    def binWidths(self):
        return self._bin_widths

    @property
    def errorBar(self):
        return self.rawData[:,2]

    def manual_binning(self):
        
        bin_edges = []
        wl_grid = self.wavelengthGrid

        bin_edges.append(wl_grid[0]-(wl_grid[1]-wl_grid[0])/2)
        for i in range(wl_grid.shape[0]-1):
            bin_edges.append(wl_grid[i]+(wl_grid[i+1]-wl_grid[i])/2.0)
        bin_edges.append((wl_grid[-1]-wl_grid[-2])/2.0 + wl_grid[-1])
        self._bin_edges = np.array(bin_edges)
        self._bin_widths = np.abs(np.diff(self._bin_edges))


        
        