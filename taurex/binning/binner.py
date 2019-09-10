from taurex.log import Logger


class Binner(Logger):

    def __init__(self):
        Logger.__init__(self, self.__class__.__name__)

    def bindown(self, wngrid, spectrum, grid_width=None, error=None):
        raise NotImplementedError


    def generate_spectrum_output(self,model_output):
        pass