import pytest
import numpy as np

def gen_opacity(T, P, grid_size, mol_name):
    opacity = {'t': T,
               'p': P,
               'name': mol_name,
               'wno': np.linspace(0, 10000, grid_size),
               'xsecarr': np.random.rand(T.shape[0], P.shape[0],
                                         grid_size)}
    return opacity


def gen_ktables(T, P, ngauss, grid_size, mol_name):
    opacity = {'t': T,
               'p': P,
               'name': mol_name,
               'wno': np.linspace(0, 10000, grid_size),
               'weights': np.polynomial.legendre.leggauss(ngauss)[1]/2,
               'xsecarr': np.random.rand(T.shape[0], P.shape[0],
                                         grid_size, ngauss)}
    return opacity

