from taurex.log import Logger
from taurex.data.fittable import fitparam,Fittable
import numpy as np


class Contribution(Fittable,Logger):



    def __init__(self,name):
        Logger.__init__(self,name)
        Fittable.__init__(self)

    

    def contribute(self,model,layer):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError
    
    def prepare(self,model):
        raise NotImplementedError
        