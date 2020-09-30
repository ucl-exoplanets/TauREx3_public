from taurex.chemistry import PowerGas
from ..strategies import molecule_vmr, TPs
from hypothesis import given
from hypothesis.strategies import integers, floats
import numpy as np


@given(molecule=molecule_vmr(), alpha=floats(allow_nan=False),
                    beta=floats(allow_nan=False), gamma=floats(allow_nan=False), tp=TPs())
def test_powergas_all(molecule, alpha, beta, gamma, tp):
    
    T,P,n = tp
    mol, vmr = molecule
    mol_name = mol[0]
    cg = PowerGas(mol[0], mix_ratio_surface=vmr, alpha=alpha, beta=beta,
                  gamma=gamma)
    cg.initialize_profile(n, T, P, None)

    params = cg.fitting_parameters()

    assert f'{mol_name}_surface' in params
    assert params[f'{mol_name}_surface'][2]() == vmr

    assert f'{mol_name}_alpha' in params
    assert params[f'{mol_name}_alpha'][2]() == alpha

    assert f'{mol_name}_beta' in params
    assert params[f'{mol_name}_beta'][2]() == beta

    assert f'{mol_name}_gamma' in params
    assert params[f'{mol_name}_gamma'][2]() == gamma