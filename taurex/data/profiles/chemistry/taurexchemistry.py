from .chemistry import Chemistry
import numpy as np
from taurex.util import molecule_texlabel
from taurex.util.util import has_duplicates
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

    def __init__(self, fill_gases=['H2', 'He'], ratio=0.17567, derived_ratios=[]):
        super().__init__('ChemistryModel')

        self._gases = []
        self._active = []
        self._inactive = []

        if isinstance(fill_gases, str):
            fill_gases = [fill_gases]

        if isinstance(ratio, float):
            ratio = [ratio]

        if has_duplicates(fill_gases):
            self.error('Fill gasses has duplicate molecules')
            self.error('Fill gasses: %s', fill_gases)
            raise ValueError('Duplicate fill gases detected')

        if len(fill_gases) > 1 and len(ratio) != len(fill_gases)-1:
            self.error('Fill gases and ratio count are not correctly matched')
            self.error('There should be %s ratios, you have defined %s',
                       len(fill_gases)-1, len(ratio))
            raise InvalidChemistryException

        self._fill_gases = fill_gases
        self._fill_ratio = ratio
        self._mix_profile = None
        self.debug('MOLECULES I HAVE %s', self.availableActive)
        self.setup_fill_params()
        self.determine_mix_mask()
        self.setup_derived_params(derived_ratios)
    
    def determine_mix_mask(self):

        try:
            self._active, self._active_mask = zip(*[(m, i) for i, m in
                                                        enumerate(self.gases)
                                                        if m in
                                                        self.availableActive])
        except ValueError:
            self.debug('No active gases detected')
            self._active, self._active_mask = [], None

        try:
            self._inactive, self._inactive_mask = zip(*[(m, i) for i, m in
                                                            enumerate(self.gases)
                                                            if m not in
                                                            self.availableActive])
        except ValueError:
            self.debug('No inactive gases detected')
            self._inactive, self._inactive_mask = [], None

        self._active_mask = np.array(self._active_mask)
        self._inactive_mask = np.array(self._inactive_mask)


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

            fget.__doc__ = f'{gas}/{main_gas} ratio (volume)'

            bounds = [1.0e-12, 0.1]

            default_fit = False
            self.add_fittable_param(param_name, param_tex, fget,
                                    fset, 'log', default_fit, bounds)

    def setup_derived_params(self, ratio_list):

        for elem_ratio in ratio_list:
            elem1, elem2 = elem_ratio.split('/')
            mol_name = '{}_{}_ratio'.format(elem1, elem2)
            param_name = mol_name
            param_tex = '{}/{}'.format(molecule_texlabel(elem1),
                                       molecule_texlabel(elem2))

            def read_mol(self, elem=elem_ratio):
                return np.mean(self.get_element_ratio(elem))


            fget = read_mol

            fget.__doc__ = f'{elem_ratio} ratio (volume)'

            compute = True
            self.add_derived_param(param_name, param_tex, fget, compute)


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
        self.mu_profile = np.zeros(shape=(nlayers,))
        if self.mixProfile is not None:
            mix_profile = self.mixProfile
            for idx, gasname in enumerate(self.gases):
                self.mu_profile += mix_profile[idx] * \
                    get_molecular_weight(gasname)

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
        if gas in self.availableActive:
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
            raise ValueError('Gas already exists')

        self.debug('Gas %s fill gas: %s', gas.molecule, self._fill_gases)
        if gas.molecule in self._fill_gases:
            self.error('Gas %s is already a fill gas: %s', gas.molecule,
                       self._fill_gases)
            raise ValueError('Gas already exists')

        self._gases.append(gas)

        self.determine_mix_mask()

    @property
    def gases(self):
        return self._fill_gases + [g.molecule for g in self._gases]

    @property
    def mixProfile(self):
        return self._mix_profile

    @property
    def activeGases(self):
        return self._active

    @property
    def inactiveGases(self):
        return self._inactive


    def compute_elements_mix(self):
        from taurex.util.util import split_molecule_elements
        element_dict = {}

        for g, m in zip(self.gases, self.mixProfile):
            avg_mix = m
            s = [], []
            if g != 'e-':
                s = split_molecule_elements(g)
            else:
                s = ['e-'], [1]
            
            for elements, count in s:
                count = int(count or '1')
                if elements not in element_dict:
                    element_dict[elements] = 0.0
                element_dict[elements] += count*avg_mix
        
        return element_dict

    
    def get_element_ratio(self, elem_ratio):
        element_dict = self.compute_elements_mix()
        elem1, elem2 = elem_ratio.split('/')

        if elem1 not in element_dict:
            self.error(f'None of the gases have the element {elem1}')
            raise ValueError(f'No gas has element {elem1}')
        if elem2 not in element_dict:
            self.error(f'None of the gases have the element {elem2}')
            raise ValueError(f'No gas has element {elem2}')
        
        return element_dict[elem1]/element_dict[elem2]







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

        mix_profile = []

        for gas in self._gases:
            gas.initialize_profile(nlayers, temperature_profile,
                                   pressure_profile, altitude_profile)
            mix_profile.append(gas.mixProfile)

        total_mix = sum(mix_profile)

        self.debug('Total mix output %s', total_mix)

        validity = np.any(total_mix > 1.0)

        self.debug('Is invalid? %s', validity)

        if validity:
            self.error('Greater than 1.0 chemistry profile detected')
            raise InvalidChemistryException

        mixratio_remainder = 1. - total_mix

        mixratio_remainder += np.zeros(shape=(nlayers))
        mix_profile = self.fill_atmosphere(mixratio_remainder) + mix_profile

        if len(mix_profile) > 0:
            self._mix_profile = np.vstack(mix_profile)
        else:
            self._mix_profile = 0.0

        super().initialize_chemistry(nlayers, temperature_profile,
                                     pressure_profile, altitude_profile)

    def fill_atmosphere(self, mixratio_remainder):

        fill = []

        if len(self._fill_gases) == 1:
            return [mixratio_remainder]
        else:
            main_molecule = mixratio_remainder*(1/(1+sum(self._fill_ratio)))

            fill.append(main_molecule)
            for molecule, ratio in zip(self._fill_gases[1:], self._fill_ratio):
                second_molecule = ratio * main_molecule
                fill.append(second_molecule)
        return fill

    @property
    def activeGasMixProfile(self):
        """
        Active gas layer by layer mix profile

        Returns
        -------
        active_mix_profile : :obj:`array`

        """
        return self.mixProfile[self._active_mask]

    @property
    def inactiveGasMixProfile(self):
        """
        Inactive gas layer by layer mix profile

        Returns
        -------
        inactive_mix_profile : :obj:`array`

        """
        return self.mixProfile[self._inactive_mask]

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

    @classmethod
    def input_keywords(cls):
        return ['taurex', 'free', ]
