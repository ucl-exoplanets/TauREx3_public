"""
Base contribution classes and functions for computing optical depth
"""

from taurex.log import Logger
from taurex.data.fittable import Fittable
import numpy as np
from taurex.output.writeable import Writeable
import numba


@numba.jit(nopython=True, nogil=True)
def contribute_tau(startK, endK, density_offset, sigma, density, path, nlayers,
                   ngrid, layer, tau):
    """
    Generic cross-section integration function for tau, numba-fied for
    performance.

    This has the form:

    .. math::

        \\tau_{\\lambda}(z) = \\int_{z_{0}}^{z_{1}} \\sigma(z') \\rho(z') dz',

    where :math:`z` is the layer, :math:`z_0` and :math:`z_1` are ``startK``
    and ``endK`` respectively. :math:`\\sigma` is the weighted
    cross-section ``sigma``. :math:`rho` is the ``density`` and
    :math:`dz'` is the integration path length ``path``


    Parameters
    ----------
    startK: int
        starting layer in integration

    endK: int
        last layer in integration

    density_offset: int
        Which part of the density profile to start from

    sigma: :obj:`array`
        cross-section

    density: array_like
        density profile of atmosphere

    path: array_like
        path-length or altitude gradient

    nlayers: int
        Total number of layers (unused)

    ngrid: int
        total number of grid points

    layer: int
        Which layer we currently on

    Returns
    -------
    tau : array_like
        optical depth (well almost you still need to do ``exp(-tau)`` yourself)

    """
    for k in range(startK, endK):
        _path = path[k]
        _density = density[k+density_offset]
        # for mol in range(nmols):
        for wn in range(ngrid):
            tau[layer, wn] += sigma[k+layer, wn]*_path*_density


class Contribution(Fittable, Logger, Writeable):
    """

    *Abstract class*

    The base class for modelling contributions to the optical depth.
    By default this handles contributions from cross-sections.
    If the type of contribution being implemented is a `sigma`-type
    like the form given in :func:`contribute_tau` then
    To function in Taurex3, it only requires the concrete implementation of:

    - :func:`prepare_each`

    Different forms may require reimplementing
    :func:`contribute` as well as :func:`prepare`

    Parameters
    ----------
    name : str
        Identifier of the contribution.

    """

    def __init__(self, name):
        Logger.__init__(self, name)
        Fittable.__init__(self)
        self._name = name
        self._total_contribution = None
        self._enabled = True
        self.sigma_xsec = None

    @property
    def order(self):
        """
        Computational order. Lower numbers are given
        higher priority and are computed first.

        Returns
        -------
        int:
            Order of computation

        """
        return 5

    @property
    def name(self):
        """Name of the contribution. Identifier for plots"""
        return self._name

    def contribute(self, model, start_layer, end_layer,
                   density_offset, layer, density, tau, path_length=None):
        """
        Computes an integral for a single layer for the optical depth.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            A forward model

        start_layer: int
            Lowest layer limit for integration

        end_layer: int
            Upper layer limit of integration

        density_offset: int
            offset in density layer

        layer: int
            atmospheric layer being computed

        density: :obj:`array`
            density profile of atmosphere

        tau: :obj:`array`
            optical depth to store result

        path_length: :obj:`array`
            integration length

        """
        self.debug('SIGMA %s', self.sigma_xsec.shape)
        self.debug(' %s %s %s %s %s %s %s', start_layer, end_layer,
                   density_offset, layer, density, tau, self._ngrid)
        contribute_tau(start_layer, end_layer, density_offset,
                       self.sigma_xsec, density, path_length, self._nlayers,
                       self._ngrid, layer, tau)
        self.debug('DONE')

    def build(self, model):
        """
        Called during forward model build phase
        Does nothing by default

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        """
        pass

    def prepare_each(self, model, wngrid):
        """
        **Requires implementation**

        Used to prepare each component of the contribution.
        For context when the main ``taurex`` program is run
        with the option each spectra is the component for the
        contribution. For cross-section based contributions,
        the components are each molecule
        Should yield the name of the component and the component itself

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            Name of component and component itself

        """

        raise NotImplementedError

    def prepare(self, model, wngrid):
        """

        Used to prepare the contribution for the calculation.
        Called before the forward model performs the main optical depth
        calculation. Default behaviour is to loop through :func:`prepare_each`
        and sum all results into a single cross-section.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid
        """

        self._ngrid = wngrid.shape[0]
        self._nlayers = model.nLayers

        sigma_xsec = np.zeros(shape=(self._nlayers, self._ngrid))

        for gas, sigma in self.prepare_each(model, wngrid):
            self.debug('Gas %s', gas)
            self.debug('Sigma %s', sigma)
            sigma_xsec += sigma

        self.sigma_xsec = sigma_xsec
        self.debug('Final sigma is %s', self.sigma_xsec)
        self.info('Done')

    def finalize(self, model, tau):
        """
        Called in the last phase of the calculation, after the optical
        depth has be completely computed.
        """
        pass

    @property
    def sigma(self):
        return self.sigma_xsec

    def write(self, output):
        """
        Writes contribution class and arguments to file

        Parameters
        ----------
        output: :class:`~taurex.output.output.Output`
        """
        contrib = output.create_group(self.__class__.__name__)
        return contrib
