import pytest
import hypothesis
from hypothesis.strategies import integers, floats,text
import numpy as np
from ..strategies import molecules, hyp_wngrid
from hypothesis.extra.numpy import arrays

@hypothesis.given(molecules(style='normal'))
def test_molecular_weight_hyp_normal(mol):
    from taurex.util.util import calculate_weight
    k,_,v = mol
    assert calculate_weight(k) == pytest.approx(v, rel=1e-3)

@hypothesis.given(molecules(style='exomol'))
def test_molecular_weight_hyp_exomol(mol):
    from taurex.util.util import calculate_weight
    k,e,v = mol
    assert calculate_weight(e) == pytest.approx(v, rel=1e-3)
    assert calculate_weight(e) == calculate_weight(k)

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


@hypothesis.given(molecules(style='exomol'))
def test_molecule_sanitization(mol):
    from taurex.util.util import sanitize_molecule_string
    expected, exomol, mass = mol
    assert sanitize_molecule_string(exomol) == expected


def test_conversion_factor():
    from taurex.util.util import conversion_factor
    assert conversion_factor('Pa', 'bar') == 1e-5

    assert conversion_factor('m', 'um') == 1e6

    assert conversion_factor('m**2/h', 'cm**2/s') == 1e4/3600

    assert conversion_factor('erg/(cm**2*s*Hz)','Jy') == 1e23


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

@hypothesis.given(integers(10, 100))
def test_bin_edges(res):
    from taurex.util.util import compute_bin_edges, create_grid_res
    grid = create_grid_res(res, 300, 10000)
    edges, widths = compute_bin_edges(grid[:, 0])
    
    assert round(np.mean(grid[:, 0]/widths)) == res


def test_check_duplicates():
    from taurex.util.util import has_duplicates
    arr = ['Hello', 'Hello']

    assert has_duplicates(arr) is True

    arr = ['Hello', 'World']

    assert has_duplicates(arr) is False


@hypothesis.given(arr=arrays(np.float, 10,
                             elements=floats(min_value=-10.0,
                                             max_value=20.0,
                                             allow_nan=False),
                             unique=True).map(np.sort),
                  value=floats(min_value=-20.0, max_value=50.0,
                               allow_nan=False))
def test_closest_pair(arr, value):
    from taurex.util.util import find_closest_pair

    left, right = find_closest_pair(arr, value)

    #hypothesis.note(f'L: {left} R: {right} V: {value}')

    assert left == right-1 or left == right
    if value < arr.min():
        assert left == 0
        assert right == 1
    elif value > arr.max():
        assert left == arr.shape[0]-2
        assert right == arr.shape[0]-1
    else:
        assert value >= arr[left]
        assert value <= arr[right]

@hypothesis.given(string=text(min_size=1))
def test_ensure_string(string):
    from taurex.util.util import ensure_string_utf8

    assert string == ensure_string_utf8(string)
    assert string == ensure_string_utf8(string.encode())

