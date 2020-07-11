
import numpy as np


class KTable:
    
    @property
    def weights(self):
        raise NotImplementedError

    def opacity(self, temperature, pressure, wngrid=None):
        from scipy.interpolate import interp1d
        wngrid_filter = slice(None)
        if wngrid is not None:
            wngrid_filter = np.where((self.wavenumberGrid >= wngrid.min()) & (
                self.wavenumberGrid <= wngrid.max()))[0]
        orig = self.compute_opacity(temperature, pressure, wngrid_filter).reshape(-1, len(self.weights))

        if wngrid is None or np.array_equal(self.wavenumberGrid.take(wngrid_filter), wngrid):
            return orig
        else:
            # min_max =  (self.wavenumberGrid <= wngrid.max() ) & (self.wavenumberGrid >= wngrid.min())

            # total_bins = self.wavenumberGrid[min_max].shape[0]
            # if total_bins > wngrid.shape[0]:
            #     return np.append(np.histogram(self.wavenumberGrid,wngrid, weights=orig)[0]/np.histogram(self.wavenumberGrid,wngrid)[0],0)

            # else:
            f = interp1d(self.wavenumberGrid[wngrid_filter], orig, axis=0, copy=False, bounds_error=False,fill_value=(orig[0],orig[-1]),assume_sorted=True)
            return f(wngrid).reshape(-1, len(self.weights))
