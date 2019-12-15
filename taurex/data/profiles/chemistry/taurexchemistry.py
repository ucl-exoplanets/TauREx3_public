from .chemistry import Chemistry
import numpy as np
from taurex.util import molecule_texlabel
from taurex.cache import OpacityCache
from taurex.exceptions import InvalidModelException


class InvalidChemistryException(InvalidModelException):
    """
    Exception that is called when atmosphere mix is greater
    than unity
    """
    pass


class TaurexChemistry(Chemistry):

    """
    The standard chemical model used in Taurex. This allows for the combination
    of different mixing profiles for each molecule. Lets take an example
    profile, we want an atmosphere with a constant mixing of ``H2O`` but two
    layer mixing for ``CH4``.
    First we initialize our chemical model:

        >>> chemistry = TaurexChemistry()

    Then we can add our molecules using the :func:`addGas` method. Lets start
    with ``H2O``, since its a constant profile for all layers of the atmosphere
    we thus add
    the :class:`~taurex.data.profiles.chemistry.gas.constantgas.ConstantGas`
    object:

        >>> chemistry.addGas(ConstantGas('H2O',mix_ratio = 1e-4))

    Easy right? Now the same goes for ``CH4``, we can add the molecule into
    the chemical model by using the correct profile (in this case
    :class:`~taurex.data.profiles.chemistry.gas.twolayergas.TwoLayerGas`):

        >>> chemistry.addGas(TwoLayerGas('CH4',mix_ratio_surface=1e-4,
                                         mix_ratio_top=1e-8))

    Molecular profiles available are:
        * :class:`~taurex.data.profiles.chemistry.gas.constantgas.ConstantGas`
        * :class:`~taurex.data.profiles.chemistry.gas.twolayergas.TwoLayerGas`
        * :class:`~taurex.data.profiles.chemistry.gas.twolayergas.TwoPointGas`


    Parameters
    ----------

    fill_gases : str or :obj:`list`
        Either a single gas or list of gases to fill the atmosphere with

    ratio : float or :obj:`list`
        If a bunch of molecules are used to fill an atmosphere, whats the
        ratio between them?
        The first fill gas is considered the main one with others defined as
        ``molecule / main_molecule``


    """

    def __init__(self, fill_gases=['H2', 'He'], ratio=0.17567):
        super().__init__('ChemistryModel')

        self._gases = []
        self._active = []
        self._inactive = []

        if isinstance(fill_gases, str):
            fill_gases = [fill_gases]

        if isinstance(ratio, float):
            ratio = [ratio]

        if len(fill_gases) > 1 and len(ratio) != len(fill_gases)-1:
            self.error('Fill gases and ratio count are not correctly matched')
            self.error('There should be %s ratios, you have defined %s',
                       len(fill_gases)-1, len(ratio))
            raise InvalidChemistryException

        self._fill_gases = fill_gases
        self._fill_ratio = ratio
        self.active_mixratio_profile = None
        self.inactive_mixratio_profile = None
        self.molecules_i_have = OpacityCache().find_list_of_molecules()
        self.debug('MOLECULES I HAVE %s', self.molecules_i_have)
        self.setup_fill_params()

    def setup_fill_params(self):
        if not hasattr(self._fill_gases, '__len__') or \
                len(self._fill_gases) < 2:
            return

        main_gas = self._fill_gases[0]

        for idx, value in enumerate(zip(self._fill_gases[1:],
                                        self._fill_ratio)):
            gas, ratio = value
            mol_name = '{}_{}'.format(gas, main_gas)
            param_name = mol_name
            param_tex = '{}/{}'.format(molecule_texlabel(gas),
                                       molecule_texlabel(main_gas))

            def read_mol(self, idx=idx):
                return self._fill_ratio[idx]

            def write_mol(self, value, idx=idx):
                self._fill_ratio[idx] = value

            fget = read_mol
            fset = write_mol

            bounds = [1.0e-12, 0.1]

            default_fit = False
            self.add_fittable_param(param_name, param_tex, fget,
                                    fset, 'log', default_fit, bounds)

    def isActive(self, gas):
        """
        Determines if the gas is active or not (Whether we have cross-sections)

        Parameters
        ----------

        gas: str
            Name of molecule


        Returns
        -------
        bool:
            True if active
        """
        if gas in self.molecules_i_have:
            return True
        else:
            return False

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
            raise InvalidChemistryException

        self.debug('Gas %s fill gas: %s', gas.molecule, self._fill_gases)
        if gas.molecule in self._fill_gases:
            self.error('Gas %s is already a fill gas: %s', gas.molecule,
                       self._fill_gases)
            raise InvalidChemistryException

        self._gases.append(gas)

    @property
    def activeGases(self):
        return self._active

    @property
    def inactiveGases(self):
        return self._inactive

    def fitting_parameters(self):
        """
        Overrides the fitting parameters to return
        one with all the gas profile parameters as well

        Returns
        -------

        fit_param : :obj:`dict`

        """
        full_dict = {}
        for gas in self._gases:
            full_dict.update(gas.fitting_parameters())

        full_dict.update(self._param_dict)

        return full_dict

    def initialize_chemistry(self, nlayers=100, temperature_profile=None,
                             pressure_profile=None, altitude_profile=None):
        """

        Initializes the chemical model and computes the all gas profiles
        and the mu profile for the forward model

        """
        self.info('Initializing chemistry model')

        active_profile = []
        inactive_profile = []

        self._active = []
        self._inactive = []
        for gas in self._gases:
            gas.initialize_profile(nlayers, temperature_profile,
                                   pressure_profile, altitude_profile)
            if self.isActive(gas.molecule):
                active_profile.append(gas.mixProfile)
                self._active.append(gas.molecule)
            else:
                inactive_profile.append(gas.mixProfile)
                self._inactive.append(gas.molecule)

        total_mix = sum(active_profile) + sum(inactive_profile)

        self.debug('Total mix output %s', total_mix)

        validity = np.any(total_mix > 1.0)

        self.debug('Is invalid? %s', validity)

        if validity:
            self.error('Greater than 1.0 chemistry profile detected')
            raise InvalidChemistryException

        mixratio_remainder = 1. - total_mix

        mixratio_remainder += np.zeros(shape=(nlayers))

        remain_active, remain_inactive = \
            self.fill_atmosphere(mixratio_remainder)

        active_profile.extend(remain_active)

        inactive_profile.extend(remain_inactive)

        if len(active_profile) > 0:
            self.active_mixratio_profile = np.vstack(active_profile)
        else:
            self.active_mixratio_profile = 0.0
        if len(inactive_profile) > 0:
            self.inactive_mixratio_profile = np.vstack(inactive_profile)
        else:
            self.inactive_mixratio_profile = 0.0
        super().initialize_chemistry(nlayers, temperature_profile,
                                     pressure_profile, altitude_profile)

    def fill_atmosphere(self, mixratio_remainder):
        active_profile = []
        inactive_profile = []

        if len(self._fill_gases) == 1:
            if self.isActive(self._fill_gases[0]):
                active_profile.append(mixratio_remainder)
                self._active.append(self._fill_gases[0])
            else:
                inactive_profile.append(mixratio_remainder)
                self._inactive.append(self._fill_gases[0])
        else:
            main_molecule = mixratio_remainder/(1. + sum(self._fill_ratio))
            if self.isActive(self._fill_gases[0]):
                active_profile.append(main_molecule)
                self._active.append(self._fill_gases[0])
            else:
                inactive_profile.append(main_molecule)
                self._inactive.append(self._fill_gases[0])
            for molecule, ratio in zip(self._fill_gases[1:], self._fill_ratio):
                second_molecule = ratio * main_molecule

                if self.isActive(molecule):
                    active_profile.append(second_molecule)
                    self._active.append(molecule)
                else:
                    inactive_profile.append(second_molecule)
                    self._inactive.append(molecule)
        return active_profile, inactive_profile

    @property
    def activeGasMixProfile(self):
        """
        Active gas layer by layer mix profile

        Returns
        -------
        active_mix_profile : :obj:`array`

        """
        return self.active_mixratio_profile

    @property
    def inactiveGasMixProfile(self):
        """
        Inactive gas layer by layer mix profile

        Returns
        -------
        inactive_mix_profile : :obj:`array`

        """
        return self.inactive_mixratio_profile

    def write(self, output):
        gas_entry = super().write(output)
        if isinstance(self._fill_gases, float):
            gas_entry.write_scalar('ratio', self._fill_ratio)
        elif hasattr(self._fill_gases, '__len__'):
            gas_entry.write_array('ratio', np.array(self._fill_ratio))
        gas_entry.write_string_array('fill_gases', self._fill_gases)
        for gas in self._gases:
            gas.write(gas_entry)

        return gas_entry
