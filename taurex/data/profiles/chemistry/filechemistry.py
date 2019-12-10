from .chemistry import Chemistry
import numpy as np
from taurex.cache import OpacityCache


class ChemistryFile(Chemistry):
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

        self._mix_ratios = np.loadtxt(filename).T
        self.molecules_i_have = OpacityCache().find_list_of_molecules()

        active = []
        active_profile = []
        inactive = []
        inactive_profile = []

        for i, g in enumerate(self._gases):
            if g in self.molecules_i_have:
                active.append(g)
                active_profile.append(self._mix_ratios[i, :])
            else:
                inactive.append(g)
                inactive_profile.append(self._mix_ratios[i, :])

        self.active_gases = active
        self.inactive_gases = inactive

        self.active_mixratio_profile = np.array(active_profile)
        self.inactive_mixratio_profile = np.array(inactive_profile)

    @property
    def activeGases(self):
        """Returns names of actively absorbing molecules

        Returns
        -------

        active : :obj:`list` of str
            List of molecules

        """
        return self.active_gases

    @property
    def inactiveGases(self):
        """Returns names of molecules not actively absorbing

        Returns
        -------

        inactive : :obj:`list` of str
            List of molecules

        """
        return self.inactive_gases

    @property
    def activeGasMixProfile(self):
        """
        Layer by layer mixing ratio of ``active`` molecules

        Returns
        -------

        mix_profile :  :obj:`array`
            Array of shape ``(nactivegases,nlayers)``


        """
        return self.active_mixratio_profile

    @property
    def inactiveGasMixProfile(self):
        """
        Layer by layer mixing ratio of ``inactive`` molecules

        Returns
        -------

        mix_profile :  :obj:`array`
            Array of shape ``(ninactivegases,nlayers)``

        """
        return self.inactive_mixratio_profile

    def write(self, output):
        gas_entry = super().write(output)
        gas_entry.write_scalar('metallicity', self.ace_metallicity)
        gas_entry.write_scalar('co_ratio', self.ace_co)

        return gas_entry
