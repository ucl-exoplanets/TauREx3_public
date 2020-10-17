from taurex.model import ForwardModel
from taurex.core import fitparam, derivedparam
import numpy as np
from taurex.spectrum import BaseSpectrum
from taurex.binning import Binner

class LineModel(ForwardModel):

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self._m = 0.5
        self._c = 10.0

        self._x = np.linspace(1, 100, 50)


    @fitparam(param_name='c')
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = value

    @fitparam(param_name='m')
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self._m = value

    def model(self, wngrid=None, cutoff_grid=True):
        if wngrid is None:
            wngrid = self._x
        return wngrid, self._m*wngrid + self._c, None, None

    def build(self):
        pass
    
    def initialize_profiles(self):
        pass
    
    @property
    def chemistry(self):
        class Dummy:
            @property
            def muProfile(self):
                return [1.0]

        test = Dummy()
        return test

    @derivedparam(param_name='mplusc')
    def mplusc(self):
        return self._m + self._c

class LineObs(BaseSpectrum):

    def create_binner(self):
        """
        Creates the appropriate binning object
        """
        from taurex.binning import NativeBinner

        return NativeBinner()

    def __init__(self, m, c, N):
        self._m = m
        self._c = c
        self._x = np.linspace(1, 100, N)
        self._y = self._m*self._x + self._c
        self._yerr = 0.1+0.1*np.random.rand(N)
        self._y += self._yerr * np.random.randn(N)

    @property
    def spectrum(self):
        return self._y

    @property
    def wavenumberGrid(self):
        return self._x

    @property
    def errorBar(self):
        return self._yerr
