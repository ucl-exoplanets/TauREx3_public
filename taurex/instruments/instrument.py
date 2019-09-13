from taurex.log import Logger


class Instrument(Logger):


    def __init__(self):
        super().__init__(self.__class__.__name__)
    




    def model_noise(self, model, model_res=None, num_observations=1):
        raise NotImplementedError