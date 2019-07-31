from taurex.log import Logger
from taurex.util import get_molecular_weight,molecule_texlabel
from taurex.data.fittable import fitparam,Fittable
import numpy as np
from taurex.output.writeable import Writeable
import math

class Chemistry(Fittable,Logger,Writeable):
    """
    Skeleton for defining chemistry.

    Parameters
    ----------
    name : str
        Name used in logging

    """

    


    def __init__(self,name):
        Logger.__init__(self,name)
        Fittable.__init__(self)

        self.mu_profile = None

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
    

    def initialize_chemistry(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        """
        **Requires implementation**

        Derived classes should implement this to compute the active and inactive gas profiles

        """


        self.compute_mu_profile(nlayers)


    @property
    def activeGasMixProfile(self):
        """
        **Requires implementation**

        Should return profiles of shape ``(nactivegases,nlayers)``. Active refers
        to gases that are actively absorbing in the atmosphere. Another way to put it
        these are gases where molecular cross-sections are used

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
        Mix profile of atmopshere

        Returns
        -------
        mix_profile : :obj:`array`

        """
        return self.mu_profile


    def get_gas_mix_profile(self,gas_name):
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


    def compute_mu_profile(self,nlayers):
        self.mu_profile= np.zeros(shape=(nlayers,))
        if self.activeGasMixProfile is not None:
            for idx, gasname in enumerate(self.activeGases):
                self.mu_profile += self.activeGasMixProfile[idx]*get_molecular_weight(gasname)
        if self.inactiveGasMixProfile is not None:
            for idx, gasname in enumerate(self.inactiveGases):
                self.mu_profile += self.inactiveGasMixProfile[idx]*get_molecular_weight(gasname)


    def write(self,output):

        gas_entry = output.create_group('Chemistry')
        gas_entry.write_string('chemistry_type',self.__class__.__name__)
        gas_entry.write_string_array('active_gases',self.activeGases)
        gas_entry.write_string_array('inactive_gases',self.inactiveGases)
        #gas_entry.write_array('active_gas_mix_profile',self.activeGasMixProfile)
        #gas_entry.write_array('inactive_gas_mix_profile',self.inactiveGasMixProfile)
        #gas_entry.write_array('mu_profile',self.muProfile)
        return gas_entry
