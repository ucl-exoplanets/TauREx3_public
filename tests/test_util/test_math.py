import pytest
import numpy as np
import hypothesis
from hypothesis.strategies import floats
import hypothesis.extra.numpy as hnum


@pytest.mark.parametrize(
    "test_input,expected",
    [
            ([0.0, 1.0, 0.5, 0.0, 1.0], 0.5),
            ([0.0, 1.0, 500, 0, 1000], 0.5),

    ])
def test_lin(test_input, expected):
    from taurex.util.math import interp_lin_only

    val = interp_lin_only(*test_input)

    assert val == expected


@hypothesis.given(T=floats(0, 1), P=floats(0, 1),
                  x11=floats(1e-30, 1e-1), x12=floats(1e-30, 1e-1),
                  x21=floats(1e-30, 1e-1), x22=floats(1e-30, 1e-1))
@hypothesis.settings(deadline=None)   # This requires a compilation stage initially
def test_bilin(T, P, x11, x12, x21, x22):
    from taurex.util.math import intepr_bilin, interp_lin_only
    Pmin, Pmax = 0.0, 1.0
    Tmin, Tmax = 0.0, 1.0
    val = intepr_bilin(np.array([x11]), np.array([x12]), np.array([x21]),np.array([x22]), 
                       T, Tmin, Tmax, P, Pmin, Pmax)
    assert pytest.approx(val[0]) == interp_lin_only(interp_lin_only(x11, x12, T, Tmin, Tmax),
                                     interp_lin_only(x21, x22, T, Tmin, Tmax),
                                     P, Pmin, Pmax)


@hypothesis.given(T=floats(1, 2), P=floats(1, 2),
                  a=floats(1e-30, 1e-1), b=floats(1e-30, 1e-1),
                  c=floats(1e-30, 1e-1), d=floats(1e-30, 1e-1))
#@hypothesis.settings(deadline=400)
def test_exp_lin(T, P, a, b, c, d):
    from taurex.util.math import interp_exp_and_lin, interp_lin_only, \
        interp_exp_only
    x11 = a
    x12 = b
    x21 = c
    x22 = d
    Tmin, Tmax = 1.0, 2.0
    Pmin, Pmax = 1.0, 2.0
    val = interp_exp_and_lin(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax)

    if T == Tmin or T == Tmax:
        assert val == interp_lin_only(x11, x21, P, Pmin, Pmax)
    elif P == Pmin or P == Pmax:
        assert val == interp_exp_only(x11, x12, T, Tmin, Tmax)
    else:
        assert val == interp_exp_only(interp_lin_only(x11, x21, P, Pmin, Pmax),
                                      interp_lin_only(x12, x22, P, Pmin, Pmax),
                                      T, Tmin, Tmax)

@hypothesis.given(hnum.arrays(np.float64, hnum.array_shapes(),
                  elements=floats(0.0, 1000)))
@hypothesis.example(np.array([[0.0, 0.0]]))
def test_online_variance(s):
    from taurex.util.math import OnlineVariance
    num_values = s.shape[0]
    expected = np.std(s, axis=0)

    onv = OnlineVariance()
    for x in s:
        onv.update(x)
    
    p_var = onv.parallelVariance()
    var = onv.variance
    if num_values < 2:
        assert np.isnan(var)
        assert np.isnan(p_var)
    else:
        assert np.all(np.isclose(var, p_var))
        assert np.sqrt(var) == pytest.approx(expected, rel=1e-6)
        assert np.sqrt(p_var) == pytest.approx(expected, rel=1e-6)
    #onv = Onli