from taurex.log import Logger


class Instrument(Logger):
    """
    *Abstract class*

    Defines some method that transforms
    a spectrum and generates noise.

    """
    def __init__(self):
        super().__init__(self.__class__.__name__)

    def model_noise(self, model, model_res=None, num_observations=1):
        """
        **Requires implementation**

        For a given forward model (and optional result)
        Resample the spectrum and compute noise profile.

        Parameters
        ----------

        model: :class:`~taurex.model.model.ForwardModel`
            Forward model to pass.

        model_res: :obj:`tuple`, optional
            Result from :func:`~taurex.model.model.ForwardModel.model`

        num_observations: int, optional
            Number of observations to simulate
        """

        raise NotImplementedError
