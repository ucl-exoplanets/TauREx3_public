
from .contribution import Contribution
import numpy as np
from taurex.cache import OpacityCache


class AbsorptionContribution(Contribution):
    """
    Computes the contribution to the optical depth
    occuring from molecular absorption.
    """

    def __init__(self):
        super().__init__('Absorption')
        self._opacity_cache = OpacityCache()

    def prepare_each(self, model, wngrid):
        """
        Prepares each molecular opacity by weighting them
        by their mixing ratio in the atmosphere

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            Name of molecule and weighted opacity

        """

        self.debug('Preparing model with %s', wngrid.shape)
        self._ngrid = wngrid.shape[0]
        sigma_xsec = np.zeros(shape=(model.nLayers, wngrid.shape[0]))

        # Get the opacity cache
        self._opacity_cache = OpacityCache()
        # Loop through all active gases
        for gas in model.chemistry.activeGases:

            # Clear sigma array
            sigma_xsec[...] = 0.0

            # Get the mix ratio of the gas
            gas_mix = model.chemistry.get_gas_mix_profile(gas)
            self.info('Recomputing active gas %s opacity', gas)

            # Get the cross section object relating to the gas
            xsec = self._opacity_cache[gas]
            # Loop through the layers
            for idx_layer, tp in enumerate(zip(model.temperatureProfile,
                                               model.pressureProfile)):
                self.debug('Got index,tp %s %s', idx_layer, tp)

                temperature, pressure = tp

                # Place into the array
                sigma_xsec[idx_layer] += \
                    xsec.opacity(temperature, pressure,
                                 wngrid)*gas_mix[idx_layer]

            # Temporarily assign to master cross-section
            self.sigma_xsec = sigma_xsec
            yield gas, sigma_xsec

    @property
    def sigma(self):
        """
        Returns the fused weighted cross-section
        of all active gases
        """
        return self.sigma_xsec
