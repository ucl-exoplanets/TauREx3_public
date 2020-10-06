from taurex.chemistry import Chemistry
from taurex.exceptions import InvalidModelException
import numpy as np
from taurex.util import molecule_texlabel  # For molecule LaTex labels


class ExampleChemistry(Chemistry):
    """
    This will be an example implementation
    of a chemistry profile for taurex


    Here we'll create a chemistry profile
    that will fill the atmosphere with molecules
    based on their occupancy.

    Here, all the ratios are normalized
    so higher values will fill more of the atmosphere
    This is again arbitrary and has no physical basis.


    We will create a fitting parameter
    for each molecule given as [Mol]_frac
    so if H2O is passed a fitting parameter
    H2O_frac will be created. this will demonstrate the
    'programmatic way' of creating fitting parameters

    This class can be used in TauREx3 like so:

    [Chemistry]
    chemistry_type = custom
    python_file = /path/to/example_chemistry.py
    molecules = H2, He, H2O, CH4, NH3
    fractions = 100, 25, 10, 1, 0.1

    [Fitting]
    H2O_frac:fit = True
    H2O_frac:bounds = 10,100


    Parameters
    ----------
    molecules: list
        List of molecules to add

    fractions: list
        List of `occupancy` of the molecules

    """

    def __init__(self, molecules=[], fractions=[]):
        super().__init__(self.__class__.__name__)

        self.info('Passed molecules %s', molecules)
        self.info('Passed fractions %s', molecules)

        self._molecules = molecules
        self._fractions = np.array(fractions)

        if not self._molecules:
            # Raise an exception is nothing is passed
            self.error('No molecules in atmosphere! molecules: %s', molecules)
            raise InvalidModelException('No molecules in atmosphere!!')

        if len(self._molecules) != len(self._fractions):
            # Raise another exception if both lists do not match
            # in size
            self.error('Molecules and fractions do not match! %s != %s',
                       len(self._molecules), len(self._fractions))
            raise InvalidModelException('Molecules and fractions mismatch')

        self.setup_molecule_parameters()  # Our method to setup fitting parameters




    def initialize_chemistry(self, nlayers=100, temperature_profile=None,
                             pressure_profile=None, altitude_profile=None):
        """
        Here we can perform the actual calculation of the profiles

        """

        # We do not want any negative values.
        # We can raise an exception to prevent it from
        # running in forward model mode. 
        # During retrievals, a different effect occurs,
        # this exception will be caught during sampling
        # and will tell the sampler to AVOID regions where any of them
        # are negative

        if np.any(self._fractions < 0):
            raise InvalidModelException('Lower than zero value detected!!')
        
        # Lets also ensure that the sum is not zero
        if np.sum(self._fractions) == 0.0:
            raise InvalidModelException('Zero total sum in atmosphere')

        # Lets create two lists:
        active_list = []
        inactive_list = []

        # Normalize our fraction array
        normalize_frac = self._fractions / np.linalg.norm(self._fractions)

        # Now loop though our combined lists
        for mol, frac in zip(self._molecules, normalize_frac):

            # Lets create molecular profile through the atmosphere
            fraction_array = np.ones(nlayers) * frac 

            # Is it active?
            if mol in self.activeGases:
                # Add it to the active array
                active_list.append(fraction_array)
            else:
                # Add it to the inactive array
                inactive_list.append(fraction_array)

        # Now we can combine them into one big profile
        self._active_mix = np.stack(active_list)
        self._inactive_mix = np.stack(inactive_list)

        super().compute_mu_profile(nlayers) #  Now we can compute the mu profile


    # --------------- TauREX 3 required properties -----------------

    @property
    def activeGases(self):
        """
        This is needed by taurex you can easily do this
        by checking if the molecule is within the availableActive
        list
        """
        return [mol for mol in self._molecules if mol in self.availableActive]

    @property
    def inactiveGases(self):
        """
        This is needed by taurex you can easily do this
        by checking if the molecule is NOT within the availableActive
        """
        return [mol for mol in self._molecules if mol not in self.availableActive]

    @property
    def activeGasMixProfile(self):
        return self._active_mix

    @property
    def inactiveGasMixProfile(self):
        return self._inactive_mix



    # -------- Fitting parameters ----------

    def setup_molecule_parameters(self):
        """
        This function will setup the fitting parameters programmatically
        """

        # Loop through the molecules whilst getting the index
        for idx, molecule in enumerate(self._molecules):
            
            # Lets use fancy f-strings to create the parameter name 
            param_name = f'{molecule}_frac'

            # Lets get the latex name of the molecule
            molecule_latex = molecule_texlabel(molecule)

            # Now create the parameter latex
            param_latex = f'{molecule_latex}_fraction'

            # Now we need to create the getters and setters
            # These will read and write each element in the array
            # The index=idx is needed to make sure the correct index
            # is always used for each molecule

            def mol_getter(self, index=idx):
                return self._fractions[index]

            def mol_setter(self, value, index=idx):
                self._fractions[index] = value

            # Lets set some bounds
            bounds = [1e-12, 1000]

            # Now lets add the fitting parameter!!
            self.add_fittable_param(param_name,
                                    param_latex,
                                    mol_getter,
                                    mol_setter,
                                    "log",
                                    False,
                                    bounds)
