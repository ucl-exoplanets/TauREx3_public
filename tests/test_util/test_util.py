import pytest
import hypothesis
from hypothesis.strategies import integers
import numpy as np


def test_molecular_weight():
    from taurex.util.util import calculate_weight
    expected = {
        'NH3':	17.031,
        'AsH3':	77.945,
        'C6H6':	78.114,
        'Br2':	159.808,
        'CS2':	76.143,
        'CO':	28.010,
        'CCl4':	153.822,
        'Cl2':	70.905,
        'ClO2':	67.452,
        'C2H4':	28.054,
        'H2COCH2':	44.053,
        'HCHO':	30.026,
        'H2NNH2':	32.045,
        'H2':	2.016,
        'HBr':	80.912,
        'HCl':	36.461,
        'HCN':	27.026,
        'H2O2':	34.015,
        'H2S':	34.082,
        'CH3SH':	48.109,
        'CH3NHNH2':	46.072,
        'NO':	30.006,
        'NO2':	46.006,
        'N2O4':	92.011,
        'O3':	47.998,
        'CH3COOOH':	76.051,
        'PH3':	33.998,
        'CH3CHOCH2':	58.080,
        'SiH4':	32.117,
        'SO2':	64.065,
        'SO2F2':	102.062,
    }

    for k, v in expected.items():
        assert calculate_weight(k) == pytest.approx(v, rel=1e-3)


@hypothesis.given(integers(10, 1000))
def test_grid_res(res):
    from taurex.util.util import create_grid_res

    wn = 10000/create_grid_res(res, 10.0, 1000)[::-1,0]

    assert round(np.mean(wn/np.gradient(wn))) == res

def test_molecule_sanitization_same():
    from taurex.util.util import sanitize_molecule_string
    names = [
            'NH3',
            'AsH3',
            'C6H6',
            'Br2',
            'CS2',
            'CO',
            'CCl4',
            'Cl2',
            'ClO2',
            'C2H4',
            'H2COCH2',
            'HCHO',
            'H2NNH2',
            'H2',
            'HBr',
            'HCl',
            'HCN',
            'H2O2',
            'H2S',
            'CH3SH',
            'CH3NHNH2',
            'NO',
            'NO2',
            'N2O4',
            'O3',
            'CH3COOOH',
            'PH3',
            'CH3CHOCH2',
            'SiH4',
            'SO2',
            'SO2F2'
        ]
    for n in names:
        assert sanitize_molecule_string(n) == n

@hypothesis.given(integers(10, 1000))
def test_width_conversion(s):

    from taurex.util.util import wnwidth_to_wlwidth, create_grid_res
    res = create_grid_res(s, 0.1, 10)

    wl = res[:,0]
    wlwidths = res[:,1]

    wn = 10000/wl[::-1]
    wnwidths = wnwidth_to_wlwidth(wl, wlwidths)[::-1]
    
    np.testing.assert_array_almost_equal(wnwidth_to_wlwidth(wn, wnwidths)[::-1], wlwidths, 6)





def test_molecule_sanitization_exomol():
    from taurex.util.util import sanitize_molecule_string
    names = {
            '1H2-16O': 'H2O',
            '24Mg-1H': 'MgH',
            '1H2-16O2': 'H2O2'

    }
    for k, n in names.items():
        assert sanitize_molecule_string(k) == n


def test_conversion_factor():
    from taurex.util.util import conversion_factor
    assert conversion_factor('Pa', 'bar') == 1e-5

    assert conversion_factor('m', 'um') == 1e6

    assert conversion_factor('m**2/h', 'cm**2/s') == 1e4/3600

    assert conversion_factor('erg/cm**2/s/Hz','Jy') == 1e23


def test_wngrid_clip():
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