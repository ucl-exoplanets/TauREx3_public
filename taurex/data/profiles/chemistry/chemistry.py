from taurex.log import Logger
from taurex.util import get_molecular_weight
from taurex.data.fittable import Fittable, derivedparam
import numpy as np
from taurex.output.writeable import Writeable
from taurex.cache import OpacityCache, GlobalCache
from taurex.cache.ktablecache import KTableCache
from taurex.data.citation import Citable



class Chemistry(Fittable, Logger, Writeable, Citable):
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


        if GlobalCache()['opacity_method'] == 'ktables':
            self._avail_active = KTableCache().find_list_of_molecules()
        else:

            self._avail_active = OpacityCache().find_list_of_molecules()
        #self._avail_active = OpacityCache().find_list_of_molecules()
        deactive_list = GlobalCache()['deactive_molecules']
        if deactive_list is not None:
            self._avail_active = [k for k in self._avail_active if k not in deactive_list]

    def set_star_planet(self, star, planet):
        """

        Supplies the star and planet to chemistry
        for photochemistry reasons. Does nothing by default

        Parameters
        ----------

        star: :class:`~taurex.data.stellar.star.Star`
            A star object

        planet: :class:`~taurex.data.planet.Planet`
            A planet object


        """
        pass

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
                    self.get_molecular_mass(gasname)
        if self.inactiveGasMixProfile is not None:
            for idx, gasname in enumerate(self.inactiveGases):
                self.mu_profile += self.inactiveGasMixProfile[idx] * \
                    self.get_molecular_mass(gasname)

    @property
    def gases(self):
        return self.activeGases + self.inactiveGases

    @property
    def mixProfile(self):
        return np.concatenate((self.activeGasMixProfile,
                               self.inactiveGasMixProfile))
    
    @derivedparam(param_name='mu', param_latex='$\mu$', compute=True)
    def mu(self):
        """
        Mean molecular weight at surface (amu)
        """
        
        from taurex.constants import AMU
        return self.muProfile[0]/AMU

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
        if self.hasCondensates:
            gas_entry.write_string_array('condensates', self.condensates)
        return gas_entry

    @property
    def condensates(self):
        """
        Returns a list of condensates in the atmosphere.

        Returns
        -------
        active : :obj:`list`
            List of condensates

        """

        return []
    
    @property
    def hasCondensates(self):
        return len(self.condensates) > 0

    @property
    def condensateMixProfile(self):
        """
        **Requires implementation**

        Should return profiles of shape ``(ncondensates,nlayers)``.
        """
        if len(self.condensates) == 0:
            return None
        else:
            raise NotImplementedError


    def get_condensate_mix_profile(self, condensate_name):
        """
        Returns the mix profile of a particular condensate

        Parameters
        ----------
        condensate_name : str
            Name of condensate

        Returns
        -------
        mixprofile : :obj:`array`
            Mix profile of condensate with shape ``(nlayer)``

        """
        if condensate_name in self.condensates:
            index = self.condensates.index(condensate_name)
            return self.condensateMixProfile[index]
        else:
            raise KeyError(f'Condensate {condensate_name} not found in chemistry')


    def get_molecular_mass(self, molecule):
        from taurex.util import get_molecular_weight
        return get_molecular_weight(molecule)
