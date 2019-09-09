import numpy as np
from taurex.log import Logger


class Binner(Logger):


    def __init__(self):
        Logger.__init__(self,self.__class__.__name__)


    def bindown(self,wngrid,spectrum):
        raise NotImplementedError

    




    



