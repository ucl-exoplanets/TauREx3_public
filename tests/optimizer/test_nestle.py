from . import LineModel, LineObs
from hypothesis import given, settings, strategies as st
import pytest

@pytest.mark.slow
@given(m=st.floats(1.0, 2.0), c=st.floats(1.0, 30.0))
@settings(deadline=None)
def test_optimizer(m, c):
    from taurex.optimizer import NestleOptimizer
    lm = LineModel()
    lm.m = 1.0
    lm.c = 10.0
    lo = LineObs(m=m, c=c, N=5)
    opt = NestleOptimizer(num_live_points=5, observed=lo, model=lm)
    opt.enable_fit('m')
    opt.enable_fit('c')
    opt.enable_derived('mplusc')
    opt.set_boundary('m', [0.8*m, 1.2*m])
    opt.set_boundary('c', [0.8*c, 1.2*c])

    opt.fit()

    idx, optimized_map, optimized_median, values = next(opt.get_solution())

    opt.update_model(optimized_map)
    
    assert lm.m == pytest.approx(m, rel=0.2)
    assert lm.c == pytest.approx(c, rel=0.2)


