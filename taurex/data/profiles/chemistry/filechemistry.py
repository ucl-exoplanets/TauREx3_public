from .autochemistry import AutoChemistry
import numpy as np
from taurex.cache import OpacityCache


class ChemistryFile(AutoChemistry):
    """
    Chemistry profile read from file

    Parameters
    ----------

    gases : :obj:`list`
        Gases in file

    filename : str
        filename for mix ratios


    """

    def __init__(self, gases=None, filename=None):
        super().__init__(self.__class__.__name__)

        self._gases = gases
        self._filename = filename
        self._mix_ratios = np.loadtxt(filename).T
        self.determine_active_inactive()

    @property
    def gases(self):
        return self._gases
    
    @property
    def mixProfile(self):
        return self._mix_ratios

    def write(self, output):
        gas_entry = super().write(output)
        gas_entry.write_scalar('filename',self._filename)

        return gas_entry

    @classmethod
    def input_keywords(cls):
        return ['file', 'fromfile', ]