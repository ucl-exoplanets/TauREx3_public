from .chemistry import Chemistry
from taurex.data.fittable import fitparam
import numpy as np
import math
from taurex.util import *
from taurex.cache import OpacityCache
class TaurexChemistry(Chemistry):
    """
    The standard chemical model used in Taurex. This allows for the combination
    of different mixing profiles for each molecule. Lets take an example profile, we want
    an atmosphere with a constant mixing of ``H2O`` but two layer mixing for ``CH4``.
    First we initialize our chemical model:

        >>> chemistry = TaurexChemistry()

    Then we can add our molecules using the :func:`addGas` method. Lets start with ``H2O``, since 
    its a constant profile for all layers of the atmosphere we thus add 
    the :class:`~taurex.data.profiles.chemistry.gas.constantgas.ConstantGas`
    object:

        >>> chemistry.addGas(ConstantGas('H2O',mix_ratio = 1e-4))
    
    Easy right? Now the same goes for ``CH4``, we can add the molecule into the chemical model
    by using the correct profile (in this case
    :class:`~taurex.data.profiles.chemistry.gas.twolayergas.TwoLayerGas`):

        >>> chemistry.addGas(TwoLayerGas('CH4',mix_ratio_surface=1e-4,mix_ratio_top=1e-8))

    
    Molecular profiles available are:
        * :class:`~taurex.data.profiles.chemistry.gas.constantgas.ConstantGas`
        * :class:`~taurex.data.profiles.chemistry.gas.twolayergas.TwoLayerGas`
        * :class:`~taurex.data.profiles.chemistry.gas.twolayergas.TwoPointGas`

    

    Parameters
    ----------

    fill_gases : str or obj:`list`
        Gas or gas-pair to fill the remainder of the atmosphere
        with. Defaults to H2 and He

    ratio : float
        If a pair of fill molecules are defined, whats the ratio between them?


    """
    def __init__(self,fill_gases=['H2','He'],ratio=0.17567):
        super().__init__('ChemistryModel')


        self._gases = []
        self._active = []
        self._inactive = []

        if hasattr(fill_gases,'__len__'):
            if len(fill_gases)> 2:
                self.error('Only maximum of two fill gases allowed')
                raise Exception('Fill_gases')


        self._fill_gases = fill_gases
        self._fill_ratio = ratio
        self.active_mixratio_profile = None
        self.inactive_mixratio_profile = None
        self.molecules_i_have = OpacityCache().find_list_of_molecules()
        self.debug('MOLECULES I HAVE %s',self.molecules_i_have)
        self.setup_fill_params()

    def setup_fill_params(self):
        if not hasattr(self._fill_gases,'__len__') or len(self._fill_gases)< 2:
            return
        
        mol_name = '{}_{}'.format(*self._fill_gases)
        param_name = mol_name
        param_tex = '{}/{}'.format(molecule_texlabel(self._fill_gases[1]),molecule_texlabel(self._fill_gases[0]))
        
        def read_mol(self):
            return self._fill_ratio
        def write_mol(self,value):
            self._fill_ratio = value

        fget = read_mol
        fset = write_mol
        
        bounds = [1.0e-12, 0.1]
        
        default_fit = False
        self.add_fittable_param(param_name,param_tex,fget,fset,'log',default_fit,bounds) 


    def isActive(self,gas):
        """
        Determines if the gas is active or not (Whether we have cross-sections)
        """
        if gas in self.molecules_i_have:
            return True
        else:
            return False


    def addGas(self,gas):
        """
        Adds a gas in the atmosphere.

        Parameters
        ----------
        gas : :class:`~taurex.data.profiles.chemistry.gas.gas.Gas`
            Gas to add into the atmosphere. Only takes effect 
            on next initialization call.

        """
        if gas.molecule in [x.molecule for x in self._gases]:
            self.error('Gas already exists')
            raise Exception('Gas already exists')
        
        self._gases.append(gas)
        



    # @property
    # def ratioMoleculeProfile(self):
    #     if self._mode in ('Relative','relative',):
    #         return self._gases[self._ratio_molecule].mixRatioProfile
    #     else:
    #         return 1.0


    @property
    def activeGases(self):
        return self._active


    @property
    def inactiveGases(self):
        return  self._inactive


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

    def initialize_chemistry(self,nlayers,temperature_profile,pressure_profile,altitude_profile):
        """

        Initializes the chemical model and computes the all gas profiles
        and the mu profile for the forward model

        """
        self.info('Initializing chemistry model')

        



        #self.active_mixratio_profile = np.zeros(shape=(len(self._gases),nlayers))
        #self.inactive_mixratio_profile = np.zeros((len(self.inactiveGases), nlayers))

        active_profile = []
        inactive_profile = []

        self._active = []
        self._inactive = []
        for gas in self._gases:
            gas.initialize_profile(nlayers,temperature_profile,pressure_profile,altitude_profile)
            if self.isActive(gas.molecule):
                active_profile.append(gas.mixProfile)
                self._active.append(gas.molecule)
            else:
                inactive_profile.append(gas.mixProfile)
                self._inactive.append(gas.molecule)


        total_mix = sum(active_profile) + sum(inactive_profile)
        


        mixratio_remainder = 1. - total_mix
        if isinstance(self._fill_gases,str) or len(self._fill_gases) ==1:
            #Simple, only one molecule so use that
            if self.isActive(self._fill_gases):
                active_profile.append(mixratio_remainder)
                self._active.append(self._fill_gases)
            else:
                inactive_profile.append(mixratio_remainder)
                self._inactive.append(self._fill_gases)
        else:
            first_pair = mixratio_remainder/(1. + self._fill_ratio) # H2
            second_pair =  self._fill_ratio * first_pair
            if self.isActive(self._fill_gases[0]):
                active_profile.append(first_pair)
                self._active.append(self._fill_gases[0])
            else:
                inactive_profile.append(first_pair)
                self._inactive.append(self._fill_gases[0])
            if self.isActive(self._fill_gases[1]):
                active_profile.append(second_pair)
                self._active.append(self._fill_gases[1])
            else:
                inactive_profile.append(second_pair)
                self._inactive.append(self._fill_gases[1])


        if len(active_profile) > 0:
            self.active_mixratio_profile = np.vstack(active_profile)
        else:
            self.active_mixratio_profile = 0.0
        if len(inactive_profile) > 0:
            self.inactive_mixratio_profile = np.vstack(inactive_profile)
        else:
            self.inactive_mixratio_profile = 0.0
        super().initialize_chemistry(nlayers,temperature_profile,pressure_profile,altitude_profile)
        


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



    def write(self,output):
        gas_entry = super().write(output)
        gas_entry.write_scalar('n2_mix_ratio',self._n2_mix_ratio)
        gas_entry.write_scalar('he_h2_ratio',self._he_h2_mix_ratio)
        for gas in self._gases:
            gas.write(gas_entry)

        return gas_entry

        