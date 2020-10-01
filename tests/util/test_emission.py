import astropy.units as u
from astropy.modeling.models import BlackBody
from taurex.util.emission import black_body, black_body_numpy
import numpy as np
import pytest
import hypothesis
from hypothesis.strategies import floats


@hypothesis.given(floats(100, 10000))
def test_blackbody(temperature):
    wngrid = np.linspace(200, 30000, 200)

    bb = black_body(wngrid, temperature)/np.pi

    bb_as = BlackBody(temperature=temperature * u.K)
    expect_flux = bb_as(wngrid * u.k).to(u.W/u.m**2/u.micron/u.sr,
                                         equivalencies=u.spectral_density(
                                                     wngrid*u.k))

    assert bb == pytest.approx(expect_flux.value, rel=1e-3)

# @hypothesis.given(floats(100, 10000))
# def test_blackbody_numpy(temperature):
#     wngrid = np.linspace(200, 30000, 20)

#     bb = black_body_numpy(wngrid, temperature)/np.pi

#     bb_as = BlackBody(temperature=temperature * u.K)
#     expect_flux = bb_as(wngrid * u.k).to(u.W/u.m**2/u.micron/u.sr,
#                                          equivalencies=u.spectral_density(
#                                                      wngrid*u.k))

#     assert bb == pytest.approx(expect_flux.value, rel=1e-3)
    
