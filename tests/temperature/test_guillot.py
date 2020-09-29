import pytest
import unittest
from taurex.data.profiles.temperature import Guillot2010
from taurex.data.planet import Earth
from ..strategies import pressures, planets
from hypothesis import given, note, strategies as st
from taurex.exceptions import InvalidModelException
import numpy as np
import math


@given(T_irr=st.floats(max_value=1e5, allow_nan=False), 
       kappa_ir=st.floats(allow_nan=False),
       kappa_v1=st.floats(allow_nan=False), 
       kappa_v2=st.floats(allow_nan=False),
       alpha=st.floats(allow_nan=False), 
       T_int=st.floats(max_value=1e5, allow_nan=False),
       P=pressures(),
       planet=planets())
def test_guillot_behaviour(T_irr, kappa_ir, kappa_v1,
                           kappa_v2, alpha, T_int, P, planet):

    g = None

    if any([kappa_ir == 0.0, kappa_v1 == 0.0, kappa_v2 == 0.0,
            T_irr < 0, T_int < 0]):
        with pytest.raises(InvalidModelException):
            g = Guillot2010(T_irr, kappa_ir, kappa_v1, kappa_v2, alpha, T_int)
        return
    elif kappa_v1/kappa_ir == 0.0 or kappa_v2/kappa_ir == 0.0:
        with pytest.raises(InvalidModelException):
            g = Guillot2010(T_irr, kappa_ir, kappa_v1, kappa_v2, alpha, T_int)
        return
    else:
        g = Guillot2010(T_irr, kappa_ir, kappa_v1, kappa_v2, alpha, T_int)

    nlayers = P.shape[0]

    g.initialize_profile(nlayers=nlayers, planet=planet,
                         pressure_profile=P)

    # Test fitting params
    params = g.fitting_parameters()

    assert 'T_irr' in params
    assert 'kappa_irr' in params
    assert 'kappa_v1' in params
    assert 'kappa_v2' in params
    assert 'alpha' in params
    assert 'T_int_guillot' in params

    assert params['T_irr'][2]() == T_irr
    assert params['kappa_irr'][2]() == kappa_ir
    assert params['kappa_v1'][2]() == kappa_v1
    assert params['kappa_v2'][2]() == kappa_v2
    assert params['alpha'][2]() == alpha
    assert params['T_int_guillot'][2]() == T_int

    g.profile

    # Test zeroing behaviour
    try:
        old_value = params['kappa_irr'][2]()
        params['kappa_irr'][3](0.0)
        g.profile
        assert False
    except InvalidModelException:
        params['kappa_irr'][3](old_value)
        assert True

    try:
        old_value = params['kappa_v1'][2]()
        params['kappa_v1'][3](0.0)
        g.profile
        assert False
    except InvalidModelException:
        params['kappa_v1'][3](old_value)
        assert True

    try:
        old_value = params['kappa_v2'][2]()
        params['kappa_v2'][3](0.0)
        g.profile
        assert False
    except InvalidModelException:
        params['kappa_v2'][3](old_value)
        assert True


def test_guillot_values():
    """
    Should be a list of inputs and outputs
    """

    pass