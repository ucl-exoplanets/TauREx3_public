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

    def test_molecular_mass_calc(self):
        from taurex.util.util import calculate_weight
        mass_test_dict = {
            'C4H7NO':	85.104637,
            'C2H3O':	43.044699,
            'C3H6NO':	72.08596,
            'C10H15O':	151.225875,
            'C4H7NO':	85.104637,
            'C3H5NO':	71.07802,
            'C6H12N4O':	156.185922,
            'C6H11N4O':	155.177981,
            'C4H6N2O2':	114.102804,
            'C4H5N2O2':	113.094864,
            'C4H5NO3':	115.087565,
            'C4H4NO3':	114.079625,
            'C8H13NO3':	171.194035,
            'C8H12NO3':	170.186094,
            'C5H9O2':	101.123956,
            'C8H9O':	121.156759,
            'C8H6BrO2':	214.03587,
            'C4H9':	57.11441,
            'C5H11O':	87.140433,
            'C7H5O':	105.11426,
            'C7H7':	91.130737,
            'C7H7O':	107.130141,
            'C9H15NO':	153.221843,
            'C6H11O':	99.151169,
            'C6H11N3O2':	157.170683,
            'C6H10N3O2':	156.162742,
            'C8H6ClO2':	169.585279,
            'C5H5':	65.093383,
            'C3H5NOS':	103.142807,
            'C3H4NOS':	102.134866,
            'C10H13O2':	165.209399,
            'C6H3N2O4':	167.099264,
            'C2H5':	29.061176,
            'C15H11O2':	223.247197,
            'CHO':	29.018082,
            'C5H8N2O2':	128.129422,
            'C5H7N2O2':	127.121481,
            'C5H5NO2':	111.098896,
            'C5H7NO3':	129.114183,
            'C5H6NO3':	128.106242,
            'C2H3NO':	57.051402,
            'C7H13N3O2':	171.197301,
            'C7H12N3O2':	170.18936,
            'C6H7N3O':	137.139515,
            'C6H6N3O':	136.131574,
            'C4H7NO2':	101.104042,
            'C4H6NO2':	100.096101,
            'C5H7NO2':	113.114778,
            'C5H6NO2':	112.106837,
            'C6H11NO':	113.157872,
            'C14H21O2':	221.315868,
            'C6H11NO':	113.157872,
            'C6H12N2O':	128.172516,
            'C6H11N2O':	127.164575,
            'C15H15O2':	227.27896,
            'CH3':	15.034558,
            'C8H9':	105.157354,
            'C8H9O':	121.156759,
            'C5H9NOS':	131.196042,
            'C20H17O':	273.349116,
            'C14H19O3S':	267.364179,
            'C10H13O3S':	213.273591,
            'C9H11O2S':	183.247568,
            'C20H17':	257.349711,
            'C6H11NO':	113.157872,
            'C5H3N2O2S':	155.154505,
            'C5H9NO':	99.131254,
            'C20H26NO3':	328.426096,
            'C5H10N2O':	114.145898,
            'C5H9N2O':	113.137958,
            'C13H17O3S':	253.337561,
            'C5H9NOS':	131.196042,
            'C5H8NOS':	130.188101,
            'C6H5':	77.104119,
            'C9H9NO':	147.174198,
            'C9H8ClNO':	181.619195,
            'C8H7NO':	133.147581,
            'C14H19O3S':	267.364179,
            'C5H7NO':	97.115373,
            'C5H5NO2':	111.098896,
            'C3H5NO':	71.07802,
            'C3H5NO2':	87.077425,
            'C3H4NO2':	86.069484,
            'C8H15NO2':	157.210512,
            'C8H14NO2':	156.202571,
            'C6H12NO':	114.165813,
            'C6H15Si':	115.269025,
            'C4H9':	57.11441,
            'C4H9O':	73.113815,
            'C4H9S':	89.179198,
            'C2F3O':	97.016086,
            'C7H7NOS':	153.201632,
            'C4H7NO2':	101.104042,
            'C4H6NO2':	100.096101,
            'C9H21Si':	157.348878,
            'C3H9Si':	73.189173,
            'C7H7O2S':	155.194334,
            'C11H10N2O':	186.210314,
            'C11H9N2O':	185.202373,
            'C19H15':	243.323093,
            'C9H9NO2':	163.173603,
            'C9H8NO2':	162.165662,
            'C5H9NO':	99.131254,
            'C5H9NO2':	115.130659,
            'C5H8NO2':	114.122719,
            'C13H9O':	181.210438,
        }

        for k, v in mass_test_dict.items():
            mass = calculate_weight(k)
            self.assertAlmostEqual(mass, v, 1)

