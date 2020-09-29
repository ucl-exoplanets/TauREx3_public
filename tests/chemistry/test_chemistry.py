import pytest
from taurex.chemistry import Chemistry
from hypothesis import given
from hypothesis.strategies import lists
from ..strategies import molecules
from unittest.mock import patch


def setup_active(active_molecules, inactive_molecules):
    from taurex.cache import OpacityCache
    from taurex.cache.ktablecache import KTableCache
    with patch.object(OpacityCache, "find_list_of_molecules") \
            as mock_my_method_xsec:
        with patch.object(KTableCache, "find_list_of_molecules") \
                as mock_my_method_ktab:
            mock_my_method_xsec.return_value = active_molecules
            mock_my_method_ktab.return_value = inactive_molecules
            c = Chemistry('test')

    return c


@given(mols=lists(molecules()))
def test_chemistry_active_default(mols):
    from taurex.cache import GlobalCache
    active_molecules = [m[0] for m in mols]
    inactive_molecules = []
    num_molecules = len(active_molecules)

    if num_molecules > 1:
        active_molecules = active_molecules[:num_molecules//2]
        inactive_molecules = active_molecules[num_molecules//2:]

    gc = GlobalCache()
    if 'opacity_method' in gc.variable_dict:
        del gc.variable_dict['opacity_method']

    c = setup_active(active_molecules, inactive_molecules)

    if len(mols) == 0:
        assert len(c.availableActive) == 0
    else:

        assert c.availableActive == active_molecules

@given(mols=lists(molecules()))
def test_chemistry_active_xsec(mols):
    from taurex.cache import GlobalCache
    active_molecules = [m[0] for m in mols]
    inactive_molecules = []
    num_molecules = len(active_molecules)

    if num_molecules > 1:
        active_molecules = active_molecules[:num_molecules//2]
        inactive_molecules = active_molecules[num_molecules//2:]

    gc = GlobalCache()
    gc['opacity_method'] = 'xsec'

    c = setup_active(active_molecules, inactive_molecules)

    if len(mols) == 0:
        assert len(c.availableActive) == 0
    else:

        assert c.availableActive == active_molecules

@given(mols=lists(molecules()))
def test_chemistry_active_ktable(mols):
    from taurex.cache import GlobalCache
    active_molecules = [m[0] for m in mols]
    inactive_molecules = []
    num_molecules = len(active_molecules)

    if num_molecules > 1:
        active_molecules = active_molecules[:num_molecules//2]
        inactive_molecules = active_molecules[num_molecules//2:]

    gc = GlobalCache()
    gc['opacity_method'] = 'ktables'

    c = setup_active(inactive_molecules, active_molecules)

    if len(mols) == 0:
        assert len(c.availableActive) == 0
    else:

        assert c.availableActive == active_molecules

