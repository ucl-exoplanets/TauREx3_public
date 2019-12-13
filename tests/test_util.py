import unittest
import numpy as np


class UtilTest(unittest.TestCase):

    def test_wngrid_clip(self):
        from taurex.util.util import clip_native_to_wngrid
        from taurex.binning import FluxBinner

        total_values = 1000
        wngrid = np.linspace(100, 10000, total_values)

        values = np.random.rand(total_values)

        test_grid = wngrid[(wngrid > 4000) & (wngrid < 8000)]

        fb = FluxBinner(wngrid=test_grid)

        true = fb.bindown(wngrid, values)

        clipped = clip_native_to_wngrid(wngrid, test_grid)
        interp_values = np.interp(clipped, wngrid, values)
        clipped_flux = fb.bindown(clipped, interp_values)

        np.testing.assert_array_equal(true[1], clipped_flux[1])
