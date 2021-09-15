from .chemistry import Chemistry
import numpy as np

class AutoChemistry(Chemistry):
    """
    Chemistry class that automatically seperates out
    active and inactive gases

    Has a helper function that should be called when



    Parameters
    ----------

    name: str
        Name of class

    """

    def __init__(self, name):
        super().__init__(name)

        self._active = []
        self._inactive = []  
        self._inactive_mask = None
        self._active_mask = None


    def determine_active_inactive(self):

        try:
            self._active, self._active_mask = zip(*[(m, i) for i, m in
                                                        enumerate(self.gases)
                                                        if m in
                                                        self.availableActive])
        except ValueError:
            self.debug('No active gases detected')
            self._active, self._active_mask = [], None

        try:
            self._inactive, self._inactive_mask = zip(*[(m, i) for i, m in
                                                            enumerate(self.gases)
                                                            if m not in
                                                            self.availableActive])
        except ValueError:
            self.debug('No inactive gases detected')
            self._inactive, self._inactive_mask = [], None

        if self._active_mask is not None:
            self._active_mask = np.array(self._active_mask)

        if self._inactive_mask is not None:
            self._inactive_mask = np.array(self._inactive_mask)


    def compute_mu_profile(self, nlayers):
        """
        Computes molecular weight of atmosphere
        for each layer

        Parameters
        ----------
        nlayers: int
            Number of layers
        """
        self.mu_profile = np.zeros(shape=(nlayers,))
        if self.mixProfile is not None:
            mix_profile = self.mixProfile
            for idx, gasname in enumerate(self.gases):
                self.mu_profile += mix_profile[idx] * \
                    self.get_molecular_mass(gasname)

    @property
    def gases(self):
        raise NotImplementedError

    @property
    def mixProfile(self):
        raise NotImplementedError


    @property
    def activeGases(self):
        return self._active

    @property
    def inactiveGases(self):
        return self._inactive


    @property
    def activeGasMixProfile(self):
        """
        Active gas layer by layer mix profile

        Returns
        -------
        active_mix_profile : :obj:`array`

        """

        if self.mixProfile is None:
            raise Exception('No mix profile computed.')

        if self._active_mask is None:
            return None


        return self.mixProfile[self._active_mask]

    @property
    def inactiveGasMixProfile(self):
        """
        Inactive gas layer by layer mix profile

        Returns
        -------
        inactive_mix_profile : :obj:`array`

        """
        if self.mixProfile is None:
            raise Exception('No mix profile computed.')
        if self._inactive_mask is None:
            return None
        return self.mixProfile[self._inactive_mask]

