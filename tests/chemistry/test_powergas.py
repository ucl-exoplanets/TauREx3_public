import pytest
from taurex.chemistry import PowerGas
from ..strategies import molecule_vmr, TPs
from hypothesis import given
from hypothesis.strategies import integers, floats
import numpy as np


@given(molecule=molecule_vmr(), alpha=floats(allow_nan=False),
                    beta=floats(allow_nan=False), gamma=floats(allow_nan=False), tp=TPs())
def test_powergas_all(molecule, alpha, beta, gamma, tp):

    T, P, n = tp
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

    assert cg.mixProfile.shape[0] == n


@pytest.mark.parametrize(
    "molecule",
    ['H2', 'H2O', 'TiO', 'VO', 'H-', 'Na', 'K'])
@given(TP=TPs())
def test_auto_profile(molecule, TP):
    T, P, n = TP
    ps_auto = PowerGas(molecule_name=molecule, profile_type='auto')
    ps_truth = PowerGas(molecule_name='LOL', profile_type=molecule)

    ps_auto.initialize_profile(n, T, P, None)
    ps_truth.initialize_profile(n, T, P, None)

    assert np.all(ps_truth.mixProfile == ps_auto.mixProfile)

    
@pytest.mark.parametrize(
    "molecule",
    ['H2', 'H2O', 'TiO', 'VO', 'H-', 'Na', 'K'])
@given(molecule_chosen=molecule_vmr(), alpha=floats(allow_nan=False),
                    beta=floats(allow_nan=False), gamma=floats(allow_nan=False), TP=TPs())
def test_auto_profile_override(molecule, molecule_chosen, alpha, beta, gamma, TP):
    T, P, n = TP
    mol, vmr = molecule_chosen
    mol_name = mol[0]
    ps_auto = PowerGas(molecule_name=molecule, profile_type='auto')
    ps_truth = PowerGas(molecule_name=molecule, mix_ratio_surface=vmr, alpha=alpha, beta=beta,
                  gamma=gamma, profile_type='auto')

    ps_auto.initialize_profile(n, T, P, None)
    ps_truth.initialize_profile(n, T, P, None)

    assert np.all(ps_truth.mixProfile != ps_auto.mixProfile)