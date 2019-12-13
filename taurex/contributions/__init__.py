"""
Modules that deal with computing contributions to optical depth
"""
from .contribution import Contribution, contribute_tau
from .absorption import AbsorptionContribution
from .cia import CIAContribution, contribute_cia
from .rayleigh import RayleighContribution
from .simpleclouds import SimpleCloudsContribution
from .leemie import LeeMieContribution
from .flatmie import FlatMieContribution
try:
    from .bhmie import BHMieContribution
except ImportError:
    from taurex.log.logger import root_logger
    root_logger.error('MIE could not be loaded')
