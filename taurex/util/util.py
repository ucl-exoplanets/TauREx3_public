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