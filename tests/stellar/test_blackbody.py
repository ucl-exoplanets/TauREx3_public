from hypothesis import given, settings, strategies as st
from ..strategies import hyp_wngrid


@given(temperature=st.floats(), radius=st.floats(),
       distance=st.floats(), magnitudeK=st.floats(), 
       mass=st.floats(),
       metallicity=st.floats(), wngrid=hyp_wngrid())
@settings(deadline=None)
def test_blackbody_star(temperature, radius, distance,
                        magnitudeK, mass, metallicity, wngrid):
    from taurex.stellar import BlackbodyStar

    bs = BlackbodyStar(temperature, radius, distance,
                       magnitudeK, mass, metallicity)

    bs.initialize(wngrid)

    bs.sed

    assert bs.sed.shape[0] == wngrid.shape[0]