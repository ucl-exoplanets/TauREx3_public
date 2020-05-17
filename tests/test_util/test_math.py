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


@pytest.mark.parametrize(
    "test_input,expected",
    [
            ([np.array([0.0]), 
              np.array([1.0]), 
              np.array([0.0]),
              np.array([1.0]),
             0.5, 0.0, 1.0, 0.5, 0.0, 1.0], np.array(0.5)),

    ])


def test_bilin(test_input, expected):
    from taurex.util.math import intepr_bilin

    val = intepr_bilin(*test_input)

    assert val == expected

@hypothesis.given(hnum.arrays(np.float64, hnum.array_shapes(), elements=floats(0.0, 1000)))
@hypothesis.example(np.array([[0.0, 0.0]]))
def test_online_variance(s):
    from taurex.util.math import OnlineVariance
    num_values = s.shape[0]
    expected = np.std(s, axis=0)

    onv = OnlineVariance()
    for x in s:
        onv.update(x)
    
    var = onv.parallelVariance()
    if s.shape[0] < 2:
        assert np.all(np.isnan(var))
    else:
        assert np.sqrt(var) == pytest.approx(expected, rel=1e-6)
    #onv = Onli