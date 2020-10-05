import pytest
import numpy as np
from taurex.binning import FluxBinner, SimpleBinner, Binner, NativeBinner
from .strategies import wngrid_spectra
from hypothesis import given
from hypothesis.strategies import booleans


def test_binner():
    b = Binner()
    with pytest.raises(NotImplementedError):
        b.bindown(None, None)


def test_fluxbinner(spectra):
    wngrid = np.linspace(400, 20000, 100)
    fb = FluxBinner(wngrid=wngrid)
    wn, sp, _, _ = fb.bindown(*spectra)

    assert(wngrid.shape[0] == wn.shape[0])

    assert np.mean(sp) == pytest.approx(np.mean(spectra[1]), rel=0.1)


def test_simplebinner(spectra):
    wngrid = np.linspace(400, 20000, 100)
    fb = SimpleBinner(wngrid=wngrid)
    wn, sp, _, _ = fb.bindown(*spectra)

    assert(wngrid.shape[0] == wn.shape[0])

    assert np.mean(sp) == pytest.approx(np.mean(spectra[1]), rel=0.1)


@given(wngrid_spectra(sort=booleans()))
def test_native_binner(s):
    wngrid, spectra = s

    nb = NativeBinner()
    wn, sp, _, _ = nb.bindown(wngrid, spectra)

    assert np.array_equal(wn, wngrid)
    assert np.array_equal(spectra, sp)
