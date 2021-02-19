from . import ChemistryMixin, TemperatureMixin
from taurex.core import fitparam
import numpy as np

class MakeFreeMixin(ChemistryMixin):
  
    def __init_mixin__(self):
        from taurex.chemistry import TaurexChemistry
        if isinstance(self, TaurexChemistry):
            raise ValueError('Class is already free-type')
        self._mixin_new_gas_list = []
        self.active_exist = []
        self.inactive_exist = []
        self.active_nonexist = []
        self.inactive_nonexist = []
    def addGas(self, gas):
        """
        Adds a gas in the atmosphere.

        Parameters
        ----------
        gas : :class:`~taurex.data.profiles.chemistry.gas.gas.Gas`
            Gas to add into the atmosphere. Only takes effect
            on next initialization call.

        """

        if gas.molecule in [x.molecule for x in self._mixin_new_gas_list]:
            self.error('Gas already exists %s', gas.molecule)
            raise ValueError('Gas already exists')

        self._mixin_new_gas_list.append(gas)
        self.determine_new_mix_mask()
        self.norm_factor = 1.0
    def determine_new_mix_mask(self):

        current_gases = self._mixin_new_gas_list
        self.active_exist = []
        self.inactive_exist = []
        self.active_nonexist = []
        self.inactive_nonexist = []

        current_active = super().activeGases
        current_inactive = super().inactiveGases
        for g in current_gases:
        
            if g.molecule in current_active:
                self.active_exist.append((g, current_active.index(g.molecule)))
            elif g.molecule in current_inactive:
                self.inactive_exist.append((g, current_inactive.index(g.molecule)))
            elif g.molecule in super().availableActive:
                self.active_nonexist.append(g)
            else:
                self.inactive_nonexist.append(g)

    @property
    def activeGases(self):
        return list(super().activeGases) + [g.molecule for g in self.active_nonexist]

    @property
    def inactiveGases(self):
        return list(super().inactiveGases) + [g.molecule for g in self.inactive_nonexist]
    
    @property
    def activeGasMixProfile(self):
        """
        Active gas layer by layer mix profile

        Returns
        -------
        active_mix_profile : :obj:`array`

        """
        mix_profile = super().activeGasMixProfile

        if mix_profile is not None:
            for g, idx in self.active_exist:
                mix_profile[idx] = g.mixProfile
        
        if len(self.active_nonexist) > 0:
            nonexist_profile = np.array([g.mixProfile for g in self.active_nonexist])
            if mix_profile is None:
                return nonexist_profile
            else:
                return np.concatenate((mix_profile, nonexist_profile))/self.norm_factor
        else:
            if mix_profile is None:
                return None
            return mix_profile/self.norm_factor
            

    @property
    def inactiveGasMixProfile(self):
        """
        Inactive gas layer by layer mix profile

        Returns
        -------
        inactive_mix_profile : :obj:`array`

        """
        mix_profile = super().inactiveGasMixProfile
        for g, idx in self.inactive_exist:
            mix_profile[idx] = g.mixProfile
        
        if len(self.inactive_nonexist) > 0:
            nonexist_profile = np.array([g.mixProfile for g in self.inactive_nonexist])
            return np.concatenate((mix_profile, nonexist_profile))/self.norm_factor
        else:
            return mix_profile/self.norm_factor
    

    def initialize_chemistry(self, nlayers=100, temperature_profile=None,
                             pressure_profile=None, altitude_profile=None):
        self._run = False
        super().initialize_chemistry(nlayers, temperature_profile,
                                     pressure_profile, altitude_profile)
        
        for g in self._mixin_new_gas_list:
            g.initialize_profile(nlayers, temperature_profile,
                                   pressure_profile, altitude_profile)
        self._run = True
        self.norm_factor = 1.0
        
        active_norm = 0.0
        inactive_norm = 0.0
        active_mix = self.activeGasMixProfile
        inactive_mix = self.inactiveGasMixProfile

        if active_mix is not None:
            active_norm = np.sum(self.activeGasMixProfile, axis=0)
        if inactive_mix is not None:
            inactive_norm = np.sum(self.inactiveGasMixProfile, axis=0)
        


        self.norm_factor = active_norm + inactive_norm
        self.compute_mu_profile(nlayers)
    
    def compute_mu_profile(self, nlayers):
        """
        Computes molecular weight of atmosphere
        for each layer

        Parameters
        ----------
        nlayers: int
            Number of layers
        """
        from taurex.util.util import get_molecular_weight
        if not self._run:
            return
        super().compute_mu_profile(nlayers)
        self._mu_profile = super().muProfile

        for idx, g in enumerate(reversed(self.active_nonexist)):
            self._mu_profile += get_molecular_weight(g.molecule) * \
                self.activeGasMixProfile[-idx-1]


        for idx, g in enumerate(reversed(self.inactive_nonexist)):
            self._mu_profile += get_molecular_weight(g.molecule) * \
                self.inactiveGasMixProfile[-idx-1]

    def fitting_parameters(self):
        """
        Overrides the fitting parameters to return
        one with all the gas profile parameters as well

        Returns
        -------

        fit_param : :obj:`dict`

        """
        full_dict = {}
        for gas in self._mixin_new_gas_list:
            full_dict.update(gas.fitting_parameters())

        full_dict.update(self._param_dict)

        return full_dict

    @property
    def muProfile(self):
        return self._mu_profile



    @classmethod
    def input_keywords(self):
        return ['makefree', ]

class TempScaler(TemperatureMixin):

    def __init_mixin__(self, scale_factor=1.0):
        self._scale_factor = scale_factor
    
    @fitparam(param_name='T_scale')
    def scaleFactor(self):
        return self._scale_factor

    @scaleFactor.setter
    def scaleFactor(self, value):
        self._scale_factor = value

    @property
    def profile(self):
        return super().profile*self._scale_factor

    @classmethod
    def input_keywords(self):
        return ['tempscalar', ]
