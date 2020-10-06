import pytest

def test_priors_validate():
    from taurex.util.fitting import validate_priors, MalformedPriorInput

    validate_priors('(hello)')
    validate_priors('((hello),(there))')

    with pytest.raises(MalformedPriorInput):
        validate_priors('(hello')

    with pytest.raises(MalformedPriorInput):
        validate_priors('hello)')

    with pytest.raises(MalformedPriorInput):
        validate_priors('((hello),(there)')


def test_parse_priors():
    from taurex.util.fitting import parse_priors

    name, args = parse_priors("HELLOWWORLD(var_a='this', "
                              "var_b=[10, 20, 30, 40],"
                              "var_c=1e34, var_d=False)")

    assert name == "HELLOWWORLD"
    expected = {a: b for a, b in zip(['var_a', 'var_b', 'var_c', 'var_d'],
                                     ['this', [10, 20, 30, 40], 1e34,
                                     False])}
    assert args == expected
