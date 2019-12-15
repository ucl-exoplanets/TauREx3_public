from taurex.temperature import TemperatureProfile
from taurex.core import fitparam
import numpy as np


class ExampleTemperature(TemperatureProfile):
    """
    An example implementation of a temperature profile

    Here we will compute a temperaurate like so:

        T(layer) = Ae^{damping*log(P(layer))}

    This is completely arbitrary and has no physical
    meaning but serves as an example of how one could implement it

    We will also create a fitting parameter "A_temp"
    with a log space default to fit the A_parameter
    and the "damp_factor" for the damping variable

    This profile can be loaded into taurex using the
    custom type like this:
    
    [Temperature]
    profile_type = custom
    python_file = path/to/example_temp.py
    A_param = 100
    damping = 0.3

    Parameters
    ----------

    A_param: float
        Our 'A' parameter. Will become a keyword
        in our input file

    damping: float
        Our 'damping' parameter. 
        Will become a keyword
        in our input file

    """

    def __init__(self, A_param=200.0, damping=0.3):
        super().__init__(self.__class__.__name__)

        self._A_param = A_param
        self._damping = damping

    def initialize_profile(self, planet=None, nlayers=100,
                           pressure_profile=None):
        """
        Here we create our initialize profile. This is
        called before each run of the forward model
        """
        # Always place this
        super().initialize_profile(planet, nlayers, pressure_profile)

        # Now we perform our computation
        self._temperature_array = \
            self._A_param * np.exp(self._damping * np.log10(pressure_profile))

    @property
    def profile(self):
        # Here we must return the final temperature profile
        return self._temperature_array

    # --------------Fitting parameters ----------------------

    # Here we can use the fitparam decorator
    # to create a retrievable parameter
    @fitparam(param_name="A_temp",  # A unique identifier for TauREx3
              param_latex="$A_{T}$",  # Latex form for plots
              default_bounds=[1e-20, 200],  # default bounds, always in linear space
              default_mode='log')  # By default, fit in log (unless changed by user)
    def parameterA(self):
        return self._A_param

    # We must also define a setter
    @parameterA.setter
    def parameterA(self, value):
        self._A_param = value   

    # Now a less verbose version for the damping
    @fitparam(param_name="damping_factor", param_latex="$D$",
              default_mode="linear")
    def dampingFactor(self):
        return self._damping

    @dampingFactor.setter
    def dampingFactor(self, value):
        self._damping = value
