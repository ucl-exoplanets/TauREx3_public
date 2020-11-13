from . import ChemistryMixin, TemperatureMixin
from taurex.core import fitparam


class MakeFreeMixin(ChemistryMixin):
    
    def __init_mixin__(self):
        self._gas_list = []

    def addGas(self, gas):
        """
        Adds a gas in the atmosphere.

        Parameters
        ----------
        gas : :class:`~taurex.data.profiles.chemistry.gas.gas.Gas`
            Gas to add into the atmosphere. Only takes effect
            on next initialization call.

        """

        if gas.molecule in [x.molecule for x in self._gases]:
            self.error('Gas already exists %s', gas.molecule)
            raise ValueError('Gas already exists')

        self.debug('Gas %s fill gas: %s', gas.molecule, self._fill_gases)
        if gas.molecule in self._fill_gases:
            self.error('Gas %s is already a fill gas: %s', gas.molecule,
                       self._fill_gases)
            raise ValueError('Gas already exists')

        self._gases.append(gas)

        self.determine_mix_mask()


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
