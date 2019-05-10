from taurex.constants import AMU

_mol_weight = {
    'HE':
        4.,
    'H2':
        2.,
    'N2':
        28.,
    'NA':
        23.,
    'K':
        39.0983,
    'O2':
        32.,
    'CO2':
        44.,
    'CH4':
        16.,
    'CO':
        28.01,
    'NH3':
        17.,
    'H2O':
        18.,
    'C2H2':
        26.04,
    'HCN':
        27.0253,
    'H2S':
        34.0809,
    'SIO':
        44.084,
    'SIO2':
        60.08,
    'SO':
        48.064,
    'SO2':
        64.066,
    'TIO':
        79.866,
    'VO':
        66.9409,

}

_mol_latex={

    'HE':
        'He',
    'H2':
        'H$_2$',
    'N2':
        'N$_2$',
    'O2':
        'O$_2$',
    'CO2':
        'CO$_2$',
    'CH4':
        'CH$_4$',
    'CO':
        'CO',
    'NH3':
        'NH$_3$',
    'H2O':
        'H$_2$O',
    'C2H2':
        'C$_2$H$_2$',
    'HCN':
        'HCN',
    'H2S':
        'H$_2$S',
    'SIO2':
        'SiO$_2$',
    'SO2':
        'SO$_2$',
}


def get_molecular_weight(gasname):

    gasname = gasname.upper()

    try:
        mu = _mol_weight[gasname]
    except KeyError:
        mu = 0


    return mu * AMU


def molecule_texlabel(gasname):
    gasname = gasname.upper()

    try:
        return _mol_latex[gasname]
    except KeyError:
        return gasname


def bindown(original_bin,original_data,new_bin):
    """
    This method quickly bins down by taking the mean.
    The numpy histogram function is exploited to do this quickly
    
    Parameters
    ----------
    original_bin: :obj:`numpy.array`
        The original bins for the that we want to bin down
    
    original_data: :obj:`numpy.array`
        The associated data that will be averaged along the new bins

    new_bin: :obj:`numpy.array`
        The new binnings we want to use (must have less points than the original)
    
    
    """
    import numpy as np
    return(np.histogram(original_bin, new_bin, weights=original_data)[0] /
              np.histogram(original_bin, new_bin)[0])


def movingaverage(a, n=3) :
    import numpy as np
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n