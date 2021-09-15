from .singleton import Singleton
from taurex.log import Logger


class GlobalCache(Singleton):
    """

    Allows for the storage of global variables.

    """
    def init(self):
        self.variable_dict = {}
        self.log = Logger('GlobalCache')

    def __getitem__(self, key):
        try:
            return self.variable_dict[key]
        except KeyError:
            return None

    def __setitem__(self, key, value):
        self.variable_dict[key] = value
