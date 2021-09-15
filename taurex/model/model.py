from taurex.log import Logger
from taurex.data.fittable import Fittable
from taurex.output.writeable import Writeable
from taurex.core import Citable

class ForwardModel(Logger, Fittable, Writeable, Citable):
    """A base class for producing forward models"""

    def __init__(self, name):
        Logger.__init__(self, name)
        Fittable.__init__(self)
        self.opacity_dict = {}
        self.cia_dict = {}

        self._native_grid = None

        self._derived_parameters = self.derived_parameters()
        self._fitting_parameters = self.fitting_parameters()

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

    @property
    def derivedParameters(self):
        return self._derived_parameters

    def compute_error(self,  samples, wngrid=None, binner=None):
        return {}, {}

    def write(self, output):
        model = output.create_group('ModelParameters')
        model.write_string('model_type', self.__class__.__name__)
        contrib = model.create_group('Contributions')
        for c in self.contribution_list:
            c.write(contrib)

        return model

    def generate_profiles(self):
        """
        Must return a dictionary of profiles you want to
        store after modeling
        """
        from taurex.util.output import generate_profile_dict
        if hasattr(self, 'temperatureProfile'):
            return generate_profile_dict(self)   # To ensure this change does not break anything
        else:
            return {}

    @classmethod
    def input_keywords(self):
        raise NotImplementedError

    def citations(self):
        from taurex.core import unique_citations_only
        model_citations = super().citations()
        for c in self.contribution_list:
            model_citations.extend(c.citations())

        return unique_citations_only(model_citations)
