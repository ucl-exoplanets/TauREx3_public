"""Defines constants used in taurex"""
from taurex.util.util import conversion_factor
import numpy as np
import astropy.constants as c

AMU = conversion_factor('u', 'kg')
KBOLTZ = c.k_B.value
G = c.G.value
RSOL = conversion_factor('Rsun', 'm')
RJUP = conversion_factor('Rjup', 'm')
PI = np.pi
MSOL = conversion_factor('Msun', 'kg')

MJUP = conversion_factor('Mjup', 'kg')
AU = conversion_factor('AU', 'm')
PLANCK = c.h.value
SPDLIGT = conversion_factor('c', 'm/s')


def get_constant(name, unit=None):
    from taurex.util.util import conversion_factor
    const = getattr(c, name)
    base_unit = str(const.unit)
    const = const.value
    if unit is not None:
        const *= conversion_factor(base_unit, unit)
    return const
