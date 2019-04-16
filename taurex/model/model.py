from taurex.log import Logger



class ForwardModel(Logger):
    """Base forward model class"""



    def __init__(self,atmosphere):

        self._atmosphere = atmosphere

    

     