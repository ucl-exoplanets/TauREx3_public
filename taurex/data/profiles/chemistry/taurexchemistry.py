from .chemistry import Chemistry
from taurex.external.ace import md_ace
from taurex.data.fittable import fitparam
import numpy as np
import math
class TaurexChemistry(Chemistry):


    def __init__(self):
        super().__init__('ACE')