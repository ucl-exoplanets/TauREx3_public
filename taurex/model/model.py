from taurex.log import Logger
from taurex.data.fittable import Fittable
from taurex.output.writeable import Writeable


class ForwardModel(Logger, Fittable, Writeable):
    """A base class for producing forward models"""

    def __init__(self, name):
        Logger.__init__(self, name)
        Fittable.__init__(self)
        self.opacity_dict = {}
        self.cia_dict = {}

        self._native_grid = None

        self._fitting_parameters = {}

        self.contribution_list = []

    def __getitem__(self, key):
        return self._fitting_parameters[key][2]()

    def __setitem__(self, key, value):
        return self._fitting_parameters[key][3](value)

    def defaultBinner(self):
        from taurex.binning import NativeBinner
        return NativeBinner()

    def add_contribution(self, contrib):
        from taurex.contributions import Contribution
        if not isinstance(contrib, Contribution):
            raise TypeError('Is not a a contribution type')
        else:
            if not contrib in self.contribution_list:
                self.contribution_list.append(contrib)
            else:
                raise Exception('Contribution already exists')

    def build(self):
        raise NotImplementedError

    def model(self, wngrid=None, cutoff_grid=True):
        """Computes the forward model for a wngrid"""
        raise NotImplementedError

    def model_full_contrib(self, wngrid=None, cutoff_grid=True):
        """Computes the forward model for a wngrid for each contribution"""
        raise NotImplementedError

    @property
    def fittingParameters(self):
        return self._fitting_parameters

    def compute_error(self,  samples, wngrid=None, binner=None):
        return {}, {}

    def write(self, output):
        model = output.create_group('ModelParameters')
        model.write_string('model_type', self.__class__.__name__)
        contrib = model.create_group('Contributions')
        for c in self.contribution_list:
            c.write(contrib)

        return model
