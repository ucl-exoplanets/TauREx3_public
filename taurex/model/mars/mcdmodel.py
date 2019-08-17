from taurex.log import Logger
import numpy as np
import math
from .model import ForwardModel
from taurex.util import bindown

class MarsMCDModel(ForwardModel):

    
    """
    def __init__(self,name,