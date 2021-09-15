import pytest
import numpy as np
ARRAY_SIZE = 10000
NLAYERS = 100
N_MU = 10


@pytest.fixture
def wngrid():
    yield np.linspace(300, 30000, ARRAY_SIZE)

@pytest.fixture
def T():
    yield np.linspace(1500, 1000, NLAYERS)

@pytest.mark.bench
def test_bb_numpy(benchmark, wngrid):
    from taurex.util.emission import black_body_numpy
    benchmark(black_body_numpy, wngrid, 1000)

@pytest.mark.bench
def test_bb_numexpr(benchmark, wngrid):
    from taurex.util.emission import black_body_numexpr
    black_body_numexpr(wngrid, 1000)
    benchmark(black_body_numexpr, wngrid, 1000)

@pytest.mark.bench
def test_bb_numba(benchmark, wngrid):
    from taurex.util.emission import black_body_numba
    black_body_numba(wngrid, 1000)
    benchmark(black_body_numba, wngrid, 1000)

@pytest.mark.bench
def test_bb_numba_II(benchmark, wngrid):
    from taurex.util.emission import black_body_numba_II
    benchmark(black_body_numba_II, wngrid, 1000)