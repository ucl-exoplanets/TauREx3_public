import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats, lists
from ..strategies import molecule_vmr, TPs
from taurex.chemistry import TaurexChemistry
from taurex.chemistry import ConstantGas
from taurex.exceptions import InvalidModelException

@given(ratio=floats(min_value=0.0, max_value=0.9),tp=TPs())
@settings(deadline=None)
def test_fill_gas_pair_ratio(ratio, tp):
    
    tc = TaurexChemistry(ratio=ratio)
    
    assert 'H2' in tc.gases
    assert 'He' in tc.gases
    
    T, P, n = tp

    tc.initialize_chemistry(n, T, P)

    h2 = tc.gases.index('H2')
    he = tc.gases.index('He')

    left = 1.0 - ratio

    assert tc.fitting_parameters()['He_H2'][2]() == ratio

    computed_ratio = tc.mixProfile[he]/tc.mixProfile[h2]

    assert computed_ratio[0] == pytest.approx(ratio)
    assert tc.mixProfile.shape[0] == 2
    assert tc.mixProfile.shape[1] == n

@given(mole_ratios=lists(molecule_vmr(min_range=0.0, max_range=1.0), min_size=1),
       tp=TPs())
@settings(deadline=None)
def test_multi_fill_gas(mole_ratios, tp):
    
    fill_gases = [m[0][0] for m in mole_ratios]
    ratios = [m[1] for m in mole_ratios[1:]]

    check_duplicates = len(fill_gases) != len(set(fill_gases))

    T, P, n = tp

    if check_duplicates:
        with pytest.raises(ValueError):
            tc = TaurexChemistry(fill_gases=fill_gases, ratio=ratios)
        return
    else:
        tc = TaurexChemistry(fill_gases=fill_gases, ratio=ratios)
    tc.initialize_chemistry(n, T, P)

    assert tc.gases == fill_gases

    main_gas = fill_gases[0]
    main_gas_mix = tc.mixProfile[tc.gases.index(fill_gases[0])]
    for f,r in zip(fill_gases[1:], ratios):
        assert tc.fitting_parameters()[f'{f}_{main_gas}'][2]() == r
        ind = tc.gases.index(f)
        assert tc.mixProfile[ind]/main_gas_mix == pytest.approx(r)

    assert tc.mixProfile.shape[0] == len(fill_gases)
    assert tc.mixProfile.shape[1] == n


@given(mols=lists(molecule_vmr(), min_size=1), tp=TPs())
@settings(deadline=None)
def test_constant_profile(mols, tp):

    molecule_names = [m[0][0] for m in mols]
    vmr = [m[1] for m in mols]

    tc = TaurexChemistry()
    
    successful = []
    unsucessful = []

    for mol, mix in zip(molecule_names, vmr):
        
        if mol in tc.gases or mol in ('H2', 'He',):
            with pytest.raises(ValueError):
                tc.addGas(ConstantGas(mol, mix))
                unsucessful.append(mol)
        else:
            tc.addGas(ConstantGas(mol, mix))
            successful.append((mol, mix))
    
    T, P, nlayers = tp

    # Now check the fitting params are included
    params = tc.fitting_parameters()

    if sum([v for s, v in successful]) > 1.0:
        with pytest.raises(InvalidModelException):
            tc.initialize_chemistry(nlayers, T, P)
        return
    else:
        tc.initialize_chemistry(nlayers, T, P)

    for s in unsucessful:
        assert s not in params
        assert s not in tc.gases

    for m, v in successful:
        assert m in params
        assert m in tc.gases
        ind = tc.gases.index(m)
        assert params[m][2]() == v
        assert tc.mixProfile[ind, 0] == v
