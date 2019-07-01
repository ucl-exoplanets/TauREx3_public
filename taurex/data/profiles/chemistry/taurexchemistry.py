from .chemistry import Chemistry
from taurex.data.fittable import fitparam
import numpy as np
import math
from taurex.util import *
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

    n2_mix_ratio : float
        Mix ratio of N2 in atmosphere

    he_h2_ratio : float
        Ratio between He and H2. The rest of the atmosphere
        is filled with these molecules.


    """
    def __init__(self,n2_mix_ratio=0,he_h2_ratio=0.1764):
        super().__init__('ChemistryModel')


        self._n2_mix_ratio = n2_mix_ratio
        self._he_h2_mix_ratio = he_h2_ratio

        self._gases = []
        self.active_mixratio_profile = None
        self.inactive_mixratio_profile = None

    def addGas(self,gas):
        """
        Adds a gas in the atmosphere.

        Parameters
        ----------
        gas : :class:`~taurex.data.profiles.chemistry.gas.gas.Gas`
            Gas to add into the atmosphere. Only takes effect 
            on next initialization call.

        """
        self._gases.append(gas)




    # @property
    # def ratioMoleculeProfile(self):
    #     if self._mode in ('Relative','relative',):
    #         return self._gases[self._ratio_molecule].mixRatioProfile
    #     else:
    #         return 1.0


    @property
    def activeGases(self):
        return [gas.molecule for gas in self._gases]


    @property
    def inactiveGases(self):
        return  ['H2', 'HE', 'N2']


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
        self.active_mixratio_profile = np.zeros(shape=(len(self._gases),nlayers))
        self.inactive_mixratio_profile = np.zeros((len(self.inactiveGases), nlayers))

        for idx,gas in enumerate(self._gases):
            gas.initialize_profile(nlayers,temperature_profile,pressure_profile,altitude_profile)
            self.active_mixratio_profile[idx,:] = gas.mixProfile


        


        #Since this can either be a scalar one or an array lets do it the old fashion way


        self.compute_absolute_gas_profile()
        

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


    def compute_absolute_gas_profile(self):
        """
        Fills whats left of the atmosphere with H2-He

        """
        
        self.inactive_mixratio_profile[2, :] = self._n2_mix_ratio
        # first get the sum of the mixing ratio of all active gases


        active_mixratio_sum = np.sum(self.active_mixratio_profile, axis = 0)
        
        active_mixratio_sum += self.inactive_mixratio_profile[2, :]
        


        mixratio_remainder = 1. - active_mixratio_sum
        self.inactive_mixratio_profile[0, :] = mixratio_remainder/(1. + self._he_h2_mix_ratio) # H2
        self.inactive_mixratio_profile[1, :] =  self._he_h2_mix_ratio * self.inactive_mixratio_profile[0, :] 



    @fitparam(param_name='N2',param_latex=molecule_texlabel('N2'),default_mode='log',default_fit=False,default_bounds=[1e-12,1.0])
    def N2MixRatio(self):
        """
        N2 mix ratio

        Parameters
        ----------
        value : float
            New mix ratio to set, must be between 0.0 and 1.0


        Returns
        -------
        n2_mix : float
        """
        return self._n2_mix_ratio
    
    @N2MixRatio.setter
    def N2MixRatio(self,value):
        self._n2_mix_ratio = value

    @fitparam(param_name='H2_He',param_latex=molecule_texlabel('H$_2$/He'),default_mode='log',default_fit=False,default_bounds=[1e-12,1.0])
    def H2HeMixRatio(self):
        """
        Ratio between H2 and He to fill the rest of the atmosphere. 
        H2 = 1 - ``H2HeMixRatio``
        He = ``H2HeMixRatio``


        Parameters
        ----------
        value : float
            New ratio to set, must be between 0.0 and 1.0


        Returns
        -------
        h2he_ratio : float
        """
        return self._he_h2_mix_ratio
    
    @H2HeMixRatio.setter
    def H2HeMixRatio(self,value):
        self._he_h2_mix_ratio = value


    def write(self,output):
        gas_entry = super().write(output)
        gas_entry.write_scalar('n2_mix_ratio',self._n2_mix_ratio)
        gas_entry.write_scalar('he_h2_ratio',self._he_h2_mix_ratio)
        for gas in self._gases:
            gas.write(gas_entry)

        return gas_entry

        