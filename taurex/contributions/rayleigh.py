from .contribution import Contribution
import numpy as np


class RayleighContribution(Contribution):
    """
    Computes contribution from Rayleigh scattering
    """

    def __init__(self):
        super().__init__('Rayleigh')

    def prepare_each(self, model, wngrid):
        """
        Computes the weighted opacity due to rayleigh
        scattering for any possible molecules within
        atmosphere.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            Name of scattering molecule and the weighted rayeligh opacity.


        """
        from taurex.util.scattering import rayleigh_sigma_from_name

        self._ngrid = wngrid.shape[0]
        self._nmols = 1
        self._nlayers = model.nLayers
        molecules = model.chemistry.activeGases + model.chemistry.inactiveGases

        for gasname in molecules:

            if np.max(model.chemistry.get_gas_mix_profile(gasname)) == 0.0:
                continue
            sigma = rayleigh_sigma_from_name(gasname, wngrid)

            if sigma is not None:
                final_sigma = sigma[None, :] * \
                    model.chemistry.get_gas_mix_profile(gasname)[:, None]
                self.sigma_xsec = final_sigma
                yield gasname, final_sigma
