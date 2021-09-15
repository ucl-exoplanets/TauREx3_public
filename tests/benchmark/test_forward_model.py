from taurex.cache import OpacityCache
import numpy as np
import pytest


@pytest.fixture
def opacities():
    from taurex.opacity.fakeopacity import FakeOpacity
    NUM_T = 27
    NUM_P = 20
    WN_RES = 10000
    WN_SIZE = 300, 30000

    from taurex.contributions import AbsorptionContribution
    OpacityCache().add_opacity(FakeOpacity('H2O'))
    OpacityCache().add_opacity(FakeOpacity('CH4'))
    yield AbsorptionContribution()
    OpacityCache().clear_cache()




@pytest.mark.bench
def test_transmission_model(benchmark, opacities):
    from taurex.model import TransmissionModel

    tm = TransmissionModel()
    tm.add_contribution(opacities)
    tm.build()

    benchmark(tm.model)

@pytest.mark.bench
def test_emission_model(benchmark, opacities):
    from taurex.model import EmissionModel

    tm = EmissionModel()
    tm.add_contribution(opacities)
    tm.build()

    benchmark(tm.model)
