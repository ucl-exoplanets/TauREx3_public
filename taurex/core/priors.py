import scipy.stats as stats
from taurex.log import Logger
import math
import enum


class PriorMode(enum.Enum):
    """
    Defines the type of prior space
    """
    LINEAR = 0,
    LOG = 1,


class Prior(Logger):
    """
    Defines a prior function
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self._prior_mode = PriorMode.LINEAR

    @property
    def priorMode(self):
        return self._prior_mode

    def sample(self, x):
        raise NotImplementedError

    def prior(self, value):
        if self._prior_mode is PriorMode.LINEAR:
            return value
        else:
            return 10**value

    def params(self):
        raise NotImplementedError
    
    def boundaries(self):
        raise NotImplementedError


class Uniform(Prior):

    def __init__(self, bounds=[0.0, 1.0]):
        super().__init__()
        if bounds is None:
            self.error('No bounds defined')
            raise ValueError('No bounds defined')

        self.set_bounds(bounds)

    def set_bounds(self, bounds):
        self._low_bounds = min(*bounds)
        self.debug('Lower bounds: %s', self._low_bounds)
        self._up_bounds = max(*bounds)
        self._scale = self._up_bounds - self._low_bounds
        self.debug('Scale: %s', self._scale)

    def sample(self, x):
        return stats.uniform.ppf(x, loc=self._low_bounds, scale=self._scale)

    def params(self):

        return f'Bounds = [{self._low_bounds},{self._up_bounds}]'

    def boundaries(self):

        return self._low_bounds, self._up_bounds


class LogUniform(Uniform):

    def __init__(self, bounds=[0.0, 1.0], lin_bounds=None):
        if lin_bounds is not None:
            bounds = [math.log10(x) for x in lin_bounds]
        super().__init__(bounds=bounds)
        self._prior_mode = PriorMode.LOG


class Gaussian(Prior):

    def __init__(self, mean=0.5, std=0.25):
        super().__init__()

        self._loc = mean
        self._scale = std

    def sample(self, x):
        return stats.norm.ppf(x, loc=self._loc, scale=self._scale)

    def params(self):

        return f'Mean = {self._loc} Stdev = {self._scale}'

    def boundaries(self):
        return self.sample(0.1), self.sample(0.9)
class LogGaussian(Gaussian):

    def __init__(self, mean=0.5, std=0.25,
                 lin_mean=None, lin_std=None):
        if lin_mean is not None:
            mean = math.log10(lin_mean)
        if lin_std is not None:
            std = math.log10(lin_std)
        super().__init__(mean=mean, std=std)
        self._prior_mode = PriorMode.LOG

