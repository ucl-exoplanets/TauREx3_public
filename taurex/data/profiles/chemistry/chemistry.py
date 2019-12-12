from taurex.log import Logger
from taurex.util import get_molecular_weight
from taurex.data.fittable import Fittable
import numpy as np
from taurex.output.writeable import Writeable
from taurex.cache import OpacityCache


class Chemistry(Fittable, Logger, Writeable):
    """
    *Abstract Class*

    Skeleton for defining chemistry. Must implement
    methods:

    - :func:`activeGases`
    - :func:`inactiveGases`
    - :func:`activeGasMixProfile`
    - :func:`inactiveGasMixProfile`

    *Active* are those that are actively
    absorbing in the atmosphere. In technical terms they are molecules
    that have absorption cross-sections. You can see which molecules
    are able to actively absorb by doing:
    You can find out what molecules can actively absorb by doing:

            >>> avail_active_mols = OpacityCache().find_list_of_molecules()

    Parameters
    ----------
    name : str
        Name used in logging

    """

    def __init__(self, name):
        Logger.__init__(self, name)
        Fittable.__init__(self)

        self.mu_profile = None
        self._avail_active = OpacityCache().find_list_of_molecules()

    @property
    def availableActive(self):
        """
        Returns a list of available
        actively absorbing molecules

        Returns
        -------
        molecules: :obj:`list`
            Actively absorbing molecules
        """
        return self._avail_active

    @property
    def activeGases(self):
        """
        **Requires implementation**

        Should return a list of molecule names

        Returns
        -------
        active : :obj:`list`
            List of active gases

        """
        raise NotImplementedError

    @property
    def inactiveGases(self):
        """
        **Requires implementation**

        Should return a list of molecule names

        Returns
        -------
        inactive : :obj:`list`
            List of inactive gases

        """
        raise NotImplementedError

    def initialize_chemistry(self, nlayers=100, temperature_profile=None,
                             pressure_profile=None, altitude_profile=None):
        """
        **Requires implementation**

        Derived classes should implement this to compute the active and
        inactive gas profiles

        Parameters
        ----------
        nlayers: int
            Number of layers in atmosphere

        temperature_profile: :obj:`array`
            Temperature profile in K, must have length ``nlayers``

        pressure_profile: :obj:`array`
            Pressure profile in Pa, must have length ``nlayers``

        altitude_profile: :obj:`array`
            Altitude profile in m, must have length ``nlayers``

        """
        self.compute_mu_profile(nlayers)

    @property
    def activeGasMixProfile(self):
        """
        **Requires implementation**

        Should return profiles of shape ``(nactivegases,nlayers)``. Active
        refers to gases that are actively absorbing in the atmosphere.
        Another way to put it these are gases where molecular cross-sections
        are used.

        """

        raise NotImplementedError

    @property
    def inactiveGasMixProfile(self):
        """
        **Requires implementation**

        Should return profiles of shape ``(ninactivegases,nlayers)``.
        These general refer to gases: ``H2``, ``He`` and ``N2``


        """
        raise NotImplementedError

    @property
    def muProfile(self):
        """
        Molecular weight for each layer of atmosphere


        Returns
        -------
        mix_profile : :obj:`array`

        """
        return self.mu_profile

    def get_gas_mix_profile(self, gas_name):
        """
        Returns the mix profile of a particular gas

        Parameters
        ----------
        gas_name : str
            Name of gas

        Returns
        -------
        mixprofile : :obj:`array`
            Mix profile of gas with shape ``(nlayer)``

        """
        if gas_name in self.activeGases:
            idx = self.activeGases.index(gas_name)
            return self.activeGasMixProfile[idx]
        elif gas_name in self.inactiveGases:
            idx = self.inactiveGases.index(gas_name)
            return self.inactiveGasMixProfile[idx]
        else:
            raise KeyError

    def compute_mu_profile(self, nlayers):
        """
        Computes molecular weight of atmosphere
        for each layer

        Parameters
        ----------
        nlayers: int
            Number of layers
        """

        self.mu_profile = np.zeros(shape=(nlayers,))
        if self.activeGasMixProfile is not None:
            for idx, gasname in enumerate(self.activeGases):
                self.mu_profile += self.activeGasMixProfile[idx] * \
                    get_molecular_weight(gasname)
        if self.inactiveGasMixProfile is not None:
            for idx, gasname in enumerate(self.inactiveGases):
                self.mu_profile += self.inactiveGasMixProfile[idx] * \
                    get_molecular_weight(gasname)

    def write(self, output):
        """
        Writes chemistry class and arguments to file

        Parameters
        ----------
        output: :class:`~taurex.output.output.Output`

        """
        gas_entry = output.create_group('Chemistry')
        gas_entry.write_string('chemistry_type', self.__class__.__name__)
        gas_entry.write_string_array('active_gases', self.activeGases)
        gas_entry.write_string_array('inactive_gases', self.inactiveGases)
        return gas_entry
