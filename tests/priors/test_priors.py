import pytest
from taurex.parameter.classfactory import ClassFactory

cf = ClassFactory()


@pytest.mark.parametrize(
    "test_input", list(cf.priorKlasses))
def test_priors(test_input):
    from taurex.core.priors import PriorMode

    prior = test_input()
    res = prior.sample(0.5)

    assert res >= 0.0
    assert res <= 1.0

    if prior.priorMode is PriorMode.LINEAR:
        assert prior.prior(0.5) == 0.5
    else:
        assert prior.prior(0.5) == 10**0.5