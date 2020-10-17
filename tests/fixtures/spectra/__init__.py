import pytest
import numpy as np


@pytest.fixture(params=['sorted', 'unsorted'])
def spectra(request):
    wngrid = np.linspace(300, 30000, 10000)
    spectra = np.random.rand(10000)*0.01

    if request.param == 'sorted':
        return wngrid, spectra
    else:
        idx = np.random.permutation(10000)
        return wngrid[idx], spectra[idx]
