
import pytest
import numpy as np
ARRAY_SIZE = 30000
NLAYERS = 500
N_MU = 4


@pytest.fixture
def wngrid():
    yield np.linspace(300, 30000, ARRAY_SIZE)

@pytest.fixture
def T():
    yield np.linspace(1500, 1000, NLAYERS)

@pytest.mark.bench
def test_integrate_emission(benchmark, wngrid, T):
    from taurex.util.emission import integrate_emission_layer
    from taurex.util.emission import black_body
    dtau = np.ones(shape=(1,wngrid.shape[0]))
    ltau = np.ones(shape=(1,wngrid.shape[0]))
    mu = np.ones(N_MU)
    def integrate_nlayers():
        for n in range(NLAYERS):
            integrate_emission_layer(dtau, ltau, mu, black_body(wngrid, T[n]))

    benchmark(integrate_nlayers)

@pytest.mark.bench
def test_integrate_emission_numba(benchmark, wngrid, T):
    from taurex.util.emission import integrate_emission_numba
    from taurex.util.emission import black_body
    dtau = np.ones(shape=(NLAYERS, wngrid.shape[0]))
    ltau = np.ones(shape=(NLAYERS, wngrid.shape[0]))
    mu = np.ones(N_MU)

    benchmark(integrate_emission_numba, wngrid, dtau, ltau, mu, T)