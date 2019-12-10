from .contribution import Contribution
import numpy as np
from taurex.data.fittable import fitparam


class SimpleCloudsContribution(Contribution):
    """
    Optically thick cloud deck up to a certain height

    These have the form:

    .. math::
            \\tau(\\lambda,z) =
                \\begin{cases}
                \\infty       & \\quad \\text{if } P(z) >= P_{0}\\\\
                0            & \\quad \\text{if } P(z) < P_{0}
                \\end{cases}

    Where :math:`P_{0}` is the pressure at the top of the cloud-deck


    Parameters
    ----------
    clouds_pressure: float
        Pressure at top of cloud deck


    """
    def __init__(self, clouds_pressure=1e3):
        super().__init__('SimpleClouds')
        self._cloud_pressure = clouds_pressure

    @property
    def order(self):
        return 3

    def contribute(self, model, start_layer, end_layer, density_offset, layer,
                   density, tau, path_length=None):
        tau[layer] += self.sigma_xsec[layer, :]

    def prepare_each(self, model, wngrid):
        """
        Returns an absorbing cross-section that is infinitely absorping
        up to a certain height

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            ``Clouds`` and opacity array.


        """

        contrib = np.zeros(shape=(model.nLayers, wngrid.shape[0],))
        cloud_filtr = model.pressureProfile >= self._cloud_pressure
        contrib[cloud_filtr, :] = np.inf
        self._contrib = contrib
        yield 'Clouds', self._contrib

    @fitparam(param_name='clouds_pressure',
              param_latex='$P_\mathrm{clouds}$',
              default_mode='log',
              default_fit=False, default_bounds=[1e-3, 1e6])
    def cloudsPressure(self):
        """
        Cloud top pressure in Pascal
        """
        return self._cloud_pressure

    @cloudsPressure.setter
    def cloudsPressure(self, value):
        self._cloud_pressure = value

    def write(self, output):
        contrib = super().write(output)
        contrib.write_scalar('clouds_pressure', self._cloud_pressure)
        return contrib
