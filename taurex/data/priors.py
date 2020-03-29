import numpy as np
from scipy import stats



def uniform_prior(x, bounds):
    bound_min = min(bounds)
    bound_max = max(bounds)

    return (x * (bound_max-bound_min)) + bound_min