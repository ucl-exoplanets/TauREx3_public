import pytest
import numpy as np
from taurex.binning import FluxBinner, SimpleBinner, Binner


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
