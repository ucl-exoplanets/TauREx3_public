from taurex.log import Logger


class Instrument(Logger):


    def __init__(self):
        super().__init__()




    def model_noise(self, model, num_observations=1):
        raise NotImplementedError