from . import ChemistryMixin, TemperatureMixin
from taurex.core import fitparam
class MakeFreeMixin(ChemistryMixin):
    pass

class TempScaler(TemperatureMixin):

    def __init__(self, scale_factor=1.0):
        super().__init__()
        self._scale_factor = scale_factor
    
    @fitparam(param_name='T_scale')
    def scaleFactor(self):
        return self._scale_factor

    @scaleFactor.setter
    def scaleFactor(self, value):
        self._scale_factor = value

    @property
    def profile(self):
        return super().profile*self._scale_factor

    @classmethod
    def input_keywords(self):
        return ['tempscalar', ]
