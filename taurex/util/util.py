"""General utility functions"""

from taurex.constants import AMU
import math
import re
import numpy as np
mass = {
  "H":	1.00794,
  "He":	4.002602,
  "Li":	6.941,
  "Be":	9.012182,
  "B":	10.811,
  "C":	12.011,
  "N":	14.00674,
  "O":	15.9994,
  "F":	18.9984032,
  "Ne":	20.1797,
  "Na":	22.989768,
  "Mg":	24.3050,
  "Al":	26.981539,
  "Si":	28.0855,
  "P":	30.973762,
  "S":	32.066,
  "Cl":	35.4527,
  "Ar":	39.948,
  "K":	39.0983,
  "Ca":	40.078,
  "Sc":	44.955910,
  "Ti":	47.88,
  "V":	50.9415,
  "Cr":	51.9961,
  "Mn":	54.93805,
  "Fe":	55.847,
  "Co":	58.93320,
  "Ni":	58.6934,
  "Cu":	63.546,
  "Zn":	65.39,
  "Ga":	69.723,
  "Ge":	72.61,
  "As":	74.92159,
  "Se":	78.96,
  "Br":	79.904,
  "Kr":	83.80,
  "Rb":	85.4678,
  "Sr":	87.62,
  "Y":	88.90585,
  "Zr":	91.224,
  "Nb":	92.90638,
  "Mo":	95.94,
  "Tc":	98,
  "Ru":	101.07,
  "Rh":	102.90550,
  "Pd":	106.42,
  "Ag":	107.8682,
  "Cd":	112.411,
  "In":	114.82,
  "Sn":	118.710,
  "Sb":	121.757,
  "Te":	127.60,
  "I":	126.90447,
  "Xe":	131.29,
  "Cs":	132.90543,
  "Ba":	137.327,
  "La":	138.9055,
  "Ce":	140.115,
  "Pr":	140.90765,
  "Nd":	144.24,
  "Pm":	145,
  "Sm":	150.36,
  "Eu":	151.965,
  "Gd":	157.25,
  "Tb":	158.92534,
  "Dy":	162.50,
  "Ho":	164.93032,
  "Er":	167.26,
  "Tm":	168.93421,
  "Yb":	173.04,
  "Lu":	174.967,
  "Hf":	178.49,
  "Ta":	180.9479,
  "W":	183.85,
  "Re":	186.207,
  "Os":	190.2,
  "Ir":	192.22,
  "Pt":	195.08,
  "Au":	196.96654,
  "Hg":	200.59,
  "Tl":	204.3833,
  "Pb":	207.2,
  "Bi":	208.98037,
  "Po":	209,
  "At":	210,
  "Rn":	222,
  "Fr":	223,
  "Ra":	226.0254,
  "Ac":	227,
  "Th":	232.0381,
  "Pa":	213.0359,
  "U":	238.0289,
  "Np":	237.0482,
  "Pu":	244,
  "Am":	243,
  "Cm":	247,
  "Bk":	247,
  "Cf":	251,
  "Es":	252,
  "Fm":	257,
  "Md":	258,
  "No":	259,
  "Lr":	260,
  "Rf":	261,
  "Db":	262,
  "Sg":	263,
  "Bh":	262,
  "Hs":	265,
  "Mt":	266,
}

def calculate_weight(chem):
    s = re.findall('([A-Z][a-z]?)([0-9]*)', chem)
    compoundweight = 0

    for element, count in s:
        count = int(count or '1')
        compoundweight += mass[element] * count
    return compoundweight


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
"""Latex versions of molecule names"""

def get_molecular_weight(gasname):
    """
    For a given molecule return the molecular weight in atomic units
    
    Parameters
    ----------
    gasname : str
        Name of molecule
    
    Returns
    -------
    float : 
        molecular weight in amu or 0 if not found
    
    """
    mu = calculate_weight(gasname)


    return mu * AMU


def molecule_texlabel(gasname):
    """
    For a given molecule return its latex form
    
    Parameters
    ----------
    gasname : str
        Name of molecule
    
    Returns
    -------
    str : 
        Latex form of the molecule or just the passed name if not found

    
    """
    gasname = gasname

    try:
        return _mol_latex[gasname]
    except KeyError:
        return gasname


def bindown(original_bin,original_data,new_bin,last_point=None):
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
    
    Returns
    -------
    :obj:`array`
        Binned mean of ``original_data``

    
    """
    import numpy as np
    #print(original_bin.shape,original_data.shape)
    #if last_point is None:
    #    last_point = new_bin[-1]*2
    
    #calc_bin = np.append(new_bin,last_point)
    #return(np.histogram(original_bin, calc_bin, weights=original_data)[0] /
    #          np.histogram(original_bin,calc_bin)[0])


    
    filter_lhs = np.zeros(new_bin.shape[0]+1)
    filter_lhs[0] = new_bin[0]
    filter_lhs[0] -= (new_bin[1] - new_bin[0])/2
    filter_lhs[-1] = new_bin[-1]
    filter_lhs[-1] += (new_bin[-1] - new_bin[-2])/2
    filter_lhs[1:-1] = (new_bin[1:] + new_bin[:-1])/2
    axis = len(original_data.shape)-1
    if axis:
        digitized = np.digitize(original_bin,filter_lhs,right=True)
        axis = len(original_data.shape)-1
        bin_means = [original_data[...,digitized == i].mean(axis=axis) for i in range(1, len(filter_lhs))]
        return np.column_stack(bin_means)
    return np.histogram(original_bin, filter_lhs, weights=original_data)[0]/np.histogram(original_bin,filter_lhs)[0]

def movingaverage(a, n=3) :
    """
    Computes moving average

    Parameters
    ----------
    a : :obj:`array`
        Array to compute average
    
    n : int
        Averaging window

    Returns
    -------
    :obj:`array`
        Resultant array

    """
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

    Parameters
    ----------

    x : :obj:`array`
        Input array or object that can be converted to an array.
    
    q : :obj:`array` or float
        Percentile or sequence of percentiles to compute, which must be between 0 and 1 inclusive.
    
    weights : :obj:`array` or float , optional
        Weights on x

    Returns
    -------
    percentile : scalar or ndarray
        

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
    """
    
    Wraps loadtext and either returns a 2d array
    or 1d array

    """

    try:
        return np.loadtxt(intext, ndmin=2)
    except:
        return np.loadtxt(intext)

def read_error_line(l):
    """ 
    Reads line?
    """
    print('_read_error_line -> line>', l)
    name, values = l.split('   ', 1)
    print('_read_error_line -> name>', name)
    print('_read_error_line -> values>', values)
    name = name.strip(': ').strip()
    values = values.strip(': ').strip()
    v, error = values.split(" +/- ")
    return name, float(v), float(error)

def read_error_into_dict(l, d):
    """
    Reads the error into dict?
    """
    name, v, error = read_error_line(l)
    d[name.lower()] = v
    d['%s error' % name.lower()] = error

def read_table(txt, d=None, title=None):
    """
    Yeah whatever i give up
    """
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

def decode_string_array(f):
    """Helper to decode strings from hdf5"""
    sl = list(f)
    return [s[0].decode('utf-8') for s in sl] 


def recursively_save_dict_contents_to_output(output, dic):
    """
    Will recursive write a dictionary into output.

    Parameters
    ----------
    output : :class:`~taurex.output.output.Output` or :class:`~taurex.output.output.OutputGroup`
        Group (or root) in output file to write to
    
    dic : :obj:`dict`
        Dictionary we want to write

    """
    import numpy as np

    for key, item in dic.items():
        
        

            try:
                store_thing(output, key, item)
            except TypeError:
                raise ValueError('Cannot save %s type'%type(item))


def store_thing(output, key, item):
    if isinstance(item, (float,int,np.int64,np.float64,)):
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
        else:
            try:
                output.write_array(key,np.array(item))
                
            except TypeError:
                for idx,val in enumerate(item):
                    new_key = '{}{}'.format(key,idx)
                    store_thing(output,new_key,val)

    elif isinstance(item, dict):
            group = output.create_group(key)
            recursively_save_dict_contents_to_output(group, item)
    else:
        raise TypeError


def weighted_avg_and_std(values, weights, axis=None):
    """
    Computes weight average and standard deviation

    Parameters
    ----------
    values : :obj:`array`
        Input array
    
    weights : :obj:`array`
        Must be same shape as ``values``
        
    
    axis : int , optional
        axis to perform weighting
    
    """


    import numpy as np
    average = np.average(values, weights=weights,axis=axis)
    variance = np.average((values-average)**2, weights=weights, axis=axis)  # Fast and numerically precise
    return (average, np.sqrt(variance))


def random_int_iter(total, fraction):
    import random
    n_points = int(total*fraction)

    samples = random.sample(range(total), n_points)
    for x in samples:
        yield x


def compute_bin_edges(wngrid):
    import numpy as np
    diff = np.diff(wngrid)/2
    edges = np.concatenate([[wngrid[0]-(wngrid[1]-wngrid[0])/2], 
                           wngrid[:-1]+diff, [(wngrid[-1]-wngrid[-2])/2 +
                           wngrid[-1]]])
    return edges, np.abs(np.diff(edges))


def clip_native_to_wngrid(native_grid, wngrid):

    min_wngrid = wngrid.min()
    max_wngrid = wngrid.max()
    #Compute the maximum width
    wnwidths = compute_bin_edges(wngrid)[-1]
    wn_min = min_wngrid - wnwidths.max()
    wn_max = max_wngrid + wnwidths.max()

    native_filter = (native_grid >= wn_min) & (native_grid <= wn_max)
    return native_grid[native_filter]

def wnwidth_to_wlwidth(wngrid, wnwidth):
    return 10000*wnwidth/(wngrid**2)



def class_for_name(module_name, class_name):
    import importlib
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c

def create_grid_res(resolution, wave_min, wave_max):
    #
    # R = l/Dl
    # l = (l-1)+Dl/2 + (Dl-1)/2
    #
    # --> (R - 1/2)*Dl = (l-1) + (Dl-1)/2
    #
    # 
    wave_list = []
    width_list = []
    wave = wave_min
    width = wave/resolution    
    
    while wave < wave_max:
        width = wave / (resolution - 0.5) + width/2/(resolution - 0.5)
        wave = resolution * width 
        width_list.append(width)
        wave_list.append(wave)

    return np.array((wave_list ,width_list)).T