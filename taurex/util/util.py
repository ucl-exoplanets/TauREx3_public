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
    #print(original_bin.shape,original_data.shape)
    calc_bin = np.append(new_bin,new_bin[-1]+0.5)
    return(np.histogram(original_bin, calc_bin, weights=original_data)[0] /
              np.histogram(original_bin,calc_bin)[0])



def movingaverage(a, n=3) :
    import numpy as np
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def quantile_corner(x, q, weights=None):
    """

    * Taken from corner.py
    __author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
    __copyright__ = "Copyright 2013-2015 Daniel Foreman-Mackey"

    Like numpy.percentile, but:

    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x

    """
    import numpy as np
    if weights is None:
        return np.percentile(x, [100. * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()

def loadtxt2d(intext):
	try:
		return np.loadtxt(intext, ndmin=2)
	except:
		return np.loadtxt(intext)

def read_error_line(l):
    print('_read_error_line -> line>', l)
    name, values = l.split('   ', 1)
    print('_read_error_line -> name>', name)
    print('_read_error_line -> values>', values)
    name = name.strip(': ').strip()
    values = values.strip(': ').strip()
    v, error = values.split(" +/- ")
    return name, float(v), float(error)

def read_error_into_dict(l, d):
    name, v, error = read_error_line(l)
    d[name.lower()] = v
    d['%s error' % name.lower()] = error

def read_table(txt, d=None, title=None):
    from io import StringIO
    import numpy as np
    if title is None:
        title, table = txt.split("\n", 1)
    else:
        table = txt
    header, table = table.split("\n", 1)
    data = loadtxt2d(StringIO(table))
    if d is not None:
        d[title.strip().lower()] = data
    if len(data.shape) == 1:
        data = np.reshape(data, (1, -1))
    return data


def recursively_save_dict_contents_to_output(output, dic):
    """
    ....
    """
    import numpy as np

    for key, item in dic.items():
        if isinstance(item,(float,int,np.int64,np.float64,)):
            output.write_scalar(key,item)
        elif isinstance(item,(np.ndarray,)):
            output.write_array(key,item)
        elif isinstance(item,(str,)):
            output.write_string(key,item)
        elif isinstance(item,(list,tuple,)):
            if isinstance(item,tuple):
                item = list(item)
            if True in [isinstance(x,str) for x in item]:
                output.write_string_array(key,item)
        
        elif isinstance(item, dict):
            group = output.create_group(key)
            recursively_save_dict_contents_to_output(group, item)
        else:
            raise ValueError('Cannot save %s type'%type(item))