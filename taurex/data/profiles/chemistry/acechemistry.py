from .chemistry import Chemistry
from taurex.external.ace import md_ace
from taurex.data.fittable import fitparam
import numpy as np
import math
from taurex.cache import OpacityCache


class ACEChemistry(Chemistry):
    """
    Equilibrium chemistry
    Computes chemical profile using the Aerotherm Chemical Equilibrium (ACE)
    Fortran code by
    Agúndez, M., Venot, O., Iro, N., et al. 2012, AandA, 548,A73

    Parameters
    ----------
    metallicity : float
        Stellar metallicity in solar units

    co_ratio : float
        C/O ratio

    therm_file : str , optional
        Location of NASA.therm file. If not set will use file included
        in library

    spec_file : str , optional
        Location of composes.dat.  If not set will use file included in library


    """

    ace_H_solar = 12.0
    """H solar abundance"""
    ace_He_solar = 10.93
    """He solar abundance"""
    ace_C_solar = 8.43
    """C solar abundance"""
    ace_O_solar = 8.69
    """O solar abundance"""
    ace_N_solar = 7.83
    """N solar abundance"""

    def __init__(self, ace_metallicity=1.0,
                 ace_co_ratio=0.54951,
                 therm_file=None,
                 spec_file=None):

        super().__init__('ACE')
        self.ace_metallicity = ace_metallicity
        self.ace_co = ace_co_ratio
        self.active_gases = None
        self.inactive_gases = None
        self._get_files(therm_file, spec_file)
        self.active_mixratio_profile = None
        self.inactive_mixratio_profile = None

    @property
    def activeGases(self):
        """
        Returns names of actively absorbing molecules

        Returns
        -------

        active : :obj:`list` of str
            List of molecules

        """
        return self.active_gases

    @property
    def inactiveGases(self):
        """
        Returns names of molecules not actively absorbing

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

    def _get_files(self, therm_file, spec_file):
        import os
        import taurex.external

        path_to_files = os.path.join(os.path.abspath(
            os.path.dirname(taurex.external.__file__)), 'ACE')
        self._specfile = spec_file
        self._thermfile = therm_file
        if self._specfile is None:
            self._specfile = os.path.join(path_to_files, 'composes.dat')
        if self._thermfile is None:
            self._thermfile = os.path.join(path_to_files, 'NASA.therm')

    def _get_gas_mask(self):
        import operator
        from taurex.constants import AMU
        self._active_mask = np.ndarray(shape=(105,), dtype=np.bool)
        self._inactive_mask = np.ndarray(shape=(105,), dtype=np.bool)
        self._active_mask[:] = False
        self._inactive_mask[:] = False
        new_active_gases = []
        new_inactive_gases = []

        self._molecule_weight = {}

        with open(self._specfile, 'r') as textfile:

            molecules_i_have = OpacityCache().find_list_of_molecules()
            self.debug('MOLECULES %s', molecules_i_have)
            for line in textfile:
                sl = line.split()
                idx = int(sl[0])-1
                molecule = sl[1]
                self._molecule_weight[molecule] = float(sl[2])*AMU

                if molecule in molecules_i_have:
                    self._active_mask[idx] = True
                    new_active_gases.append((molecule, idx))
                else:
                    self._inactive_mask[idx] = True
                    new_inactive_gases.append((molecule, idx))
        # Create a new list where the gases are in the correct order
        new_active_gases.sort(key=operator.itemgetter(1))
        new_inactive_gases.sort(key=operator.itemgetter(1))

        self.active_gases = [molecule for molecule, _ in new_active_gases]
        self.inactive_gases = [molecule for molecule, _ in new_inactive_gases]
        self.info('Active gases: %s', self.active_gases)
        self.info('Inactive gases: %s', self.inactive_gases)

    def set_ace_params(self):

        # set O, C and N abundances given metallicity (in solar units) and CO
        self.O_abund_dex = math.log10(self.ace_metallicity *
                                      (10**(self.ace_O_solar-12.)))+12.
        self.N_abund_dex = math.log10(self.ace_metallicity *
                                      (10**(self.ace_N_solar-12.)))+12.

        self.C_abund_dex = self.O_abund_dex + math.log10(self.ace_co)

        # H and He don't change
        self.H_abund_dex = self.ace_H_solar
        self.He_abund_dex = self.ace_He_solar

    def compute_active_gas_profile(self, nlayers, altitude_profile,
                                   pressure_profile,
                                   temperature_profile):
        """Computes gas profiles of both active and inactive molecules for each layer

        Parameters
        ----------

        altitude_profile : array_like
            Altitude profile of atmosphere (usually computed in model)

        pressure_profile : array_like
            Pressure profile of atmosphere

        temperature_profile : array_like
            Temperature profile of atmosphere

        """

        self._get_gas_mask()
        self.active_mixratio_profile = np.zeros(shape=(len(self.activeGases),
                                                       nlayers))
        self.inactive_mixratio_profile = np.zeros((len(self.inactiveGases),
                                                   nlayers))
        self.set_ace_params()

        # Call FORTRAN ACE function
        self._ace_profile = md_ace(self._specfile,
                                   self._thermfile,
                                   altitude_profile/1000.0,
                                   pressure_profile/1.e5,
                                   temperature_profile,
                                   self.He_abund_dex,
                                   self.C_abund_dex,
                                   self.O_abund_dex,
                                   self.N_abund_dex)

        self.active_mixratio_profile = self._ace_profile[self._active_mask, :]
        self.inactive_mixratio_profile = \
            self._ace_profile[self._inactive_mask, :]

    def initialize_chemistry(self, nlayers=100, temperature_profile=None,
                             pressure_profile=None, altitude_profile=None):
        """
        Sets up and and constructs chemical profiles. Called by forward
        model before path calculation

        Parameters
        ----------

        nlayers : int
            Number of layers in atmosphere

        altitude_profile : array_like
            Altitude profile of atmosphere (usually computed in model)

        pressure_profile : array_like
            Pressure profile of atmosphere

        temperature_profile : array_like
            Temperature profile of atmosphere

        """

        self.info('Initializing chemistry model')

        self.compute_active_gas_profile(nlayers, altitude_profile,
                                        pressure_profile, temperature_profile)

        self.compute_mu_profile(nlayers)

    @fitparam(param_name='ace_metallicity',
              param_latex='Metallicity',
              default_mode='log',
              default_fit=False,
              default_bounds=[-1, 4])
    def aceMetallicity(self):
        """
        Metallicity of star in solar units
        """
        return self.ace_metallicity

    @aceMetallicity.setter
    def aceMetallicity(self, value):
        self.ace_metallicity = value

    @fitparam(param_name='ace_co',
              param_latex='C/O',
              default_fit=False,
              default_bounds=[0, 2])
    def aceCORatio(self):
        """
        CO ratio of star
        """
        return self.ace_co

    @aceCORatio.setter
    def aceCORatio(self, value):
        self.ace_co = value

    def write(self, output):

        gas_entry = super().write(output)
        gas_entry.write_scalar('ace_metallicity', self.ace_metallicity)
        gas_entry.write_scalar('ace_co_ratio', self.ace_co)
        if self._thermfile is not None:
            gas_entry.write_string('therm_file', self._thermfile)
        if self._specfile is not None:
            gas_entry.write_string('spec_file', self._specfile)

        return gas_entry

    def compute_mu_profile(self, nlayers):
        self.mu_profile = np.zeros(shape=(nlayers,))
        if self.activeGasMixProfile is not None:
            for idx, gasname in enumerate(self.activeGases):
                self.mu_profile += self.activeGasMixProfile[idx, :] * \
                    self._molecule_weight[gasname]
        if self.inactiveGasMixProfile is not None:
            for idx, gasname in enumerate(self.inactiveGases):
                self.mu_profile += self.inactiveGasMixProfile[idx, :] * \
                    self._molecule_weight[gasname]
