from taurex.chemistry import ConstantGas
from ..strategies import molecule_vmr
from hypothesis import given
from hypothesis.strategies import integers, floats
import numpy as np


@given(molecule=molecule_vmr(), nlayers=integers(1, 300),
       new_value=floats(1e-30, 1e0))
def test_constant_gas(molecule, nlayers, new_value):

    mol, vmr = molecule
    cg = ConstantGas(mol[0], mix_ratio=vmr)
    cg.initialize_profile(nlayers=nlayers)

    mix_profile = cg.mixProfile

    assert np.all(mix_profile == vmr)

    params = cg.fitting_parameters()

    assert mol[0] in params

    params = params[mol[0]]

    name = params[0]
    getter = params[2]
    setter = params[3]

    assert name == mol[0]
    assert getter() == vmr
    setter(new_value)
    assert getter() == new_value

    cg.initialize_profile(nlayers=nlayers)
    mix_profile = cg.mixProfile
    assert np.all(mix_profile == new_value)

