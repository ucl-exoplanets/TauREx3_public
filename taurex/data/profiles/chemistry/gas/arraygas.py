from .gas import Gas
import numpy as np


class ArrayGas(Gas):
    """

    Gas profile from an array. Molecular abundance is interpolated if the
    number of layers do not match

    Parameters
    -----------
    molecule_name : str
        Name of molecule

    mix_ratio_array : :obj:`array`
        Mixing ratio of the molecule

    """

    def __init__(self, molecule_name='H2O',
                 mix_ratio_array=[1e-2, 1e-6]):
        super().__init__(self.__class__.__name__, molecule_name)

        self._mix_ratio_array = np.array(mix_ratio_array)

    @property
    def mixProfile(self):
        """

        Mixing profile

        Returns
        -------
        mix: :obj:`array`
            Mix ratio for molecule at each layer

        """

        return self._mix_array

    def initialize_profile(self, nlayers=None, temperature_profile=None,
                           pressure_profile=None, altitude_profile=None):

        if nlayers is None:
            self.error('number layers argument required')
            raise ValueError('Number of layers argument required')

        interp_array = np.linspace(0.0, 1.0, self._mix_ratio_array.shape[0])

        layer_interp = np.linspace(0.0, 1.0, nlayers)

        self._mix_array = np.interp(layer_interp,
                                    interp_array, self._mix_ratio_array)

    def write(self, output):
        """

        Parameters
        ----------
        output: :class:`~taurex.output.output.Output`
        """
        gas_entry = super().write(output)
        gas_entry.write_array('mix_ratio_array', self._mix_ratio_array)

        return gas_entry
