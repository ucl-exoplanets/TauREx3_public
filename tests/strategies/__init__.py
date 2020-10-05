import hypothesis
from hypothesis.strategies import composite, lists, tuples, integers, floats, text, booleans
from hypothesis import assume
import hypothesis.extra.numpy as hyp_numpy
import numpy as np

@composite
def molecules(draw, style='normal'):
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
    atom_names = list(mass.keys())
    total_atoms = len(mass)

    at_count_draw = tuples(integers(1, total_atoms-1), integers(1, 6))

    atoms = draw(lists(at_count_draw, min_size=1))

    name_atom_count = [(atom_names[at], count) for at,count in atoms]

    molecule_name = ''.join([f'{at}{c}' if c > 1 else f'{at}' for at, c in name_atom_count])
    exomol_name = molecule_name
    if style == 'exomol':
        exomol_name = '-'.join([f'{int(mass[at])}{at}{c}' if c > 1 else f'{int(mass[at])}{at}' for at, c in name_atom_count])
    
    final_mass = sum([mass[at]*c for at, c in name_atom_count])

    return molecule_name, exomol_name, final_mass


@composite
def molecule_vmr(draw, min_range=1e-20, max_range=1):
    molecule = draw(molecules())
    vmr = draw(floats(min_range, max_range))

    return molecule, vmr


@composite
def fitting_parameters(draw):

    fitting_names = draw(lists(text(min_size=1), min_size=1, max_size=20))
    num_fit = len(fitting_names)
    value = draw(lists(floats(1e-10, 1e10), min_size=num_fit, max_size=num_fit))
    mode = draw(lists(booleans(), min_size=num_fit, max_size=num_fit))
    mode = ['linear' if b else 'log' for b in mode]
    bounds = draw(lists(tuples(floats(1e-10, 1e10), floats(1e-10, 1e10)), min_size=num_fit, max_size=num_fit ))
    default_fit = draw(lists(booleans(), min_size=num_fit, max_size=num_fit))

    return list(zip(fitting_names, value, mode, default_fit, bounds))

@composite
def hyp_wngrid(draw, num_elements=integers(3, 50), sort=False):
    wngrid = draw(hyp_numpy.arrays(np.float64, num_elements, elements=floats(100, 30000), unique=True))
    if sort:
        wngrid = np.sort(wngrid)
    
    return wngrid


@composite
def wngrid_spectra(draw, num_elements=integers(3, 50), sort=False):

    wngrid = draw(hyp_wngrid(num_elements, sort))
    if sort:
        wngrid = np.sort(wngrid)

    spectra = draw(hyp_numpy.arrays(np.float64, wngrid.shape[0],
                                    elements=floats(1e-10, 1e-1), fill=floats(1e-10, 1e-1)))

    return wngrid, spectra


@composite
def temperatures(draw, min_layers=2, max_layers=30):
    nlayers = draw(integers(min_value=min_layers, max_value=max_layers))

    min_T = draw(floats(min_value=100.0, max_value=3000.0))
    max_T = draw(floats(min_value=100.0, max_value=3000.0))
    T = np.linspace(min_T, max_T, nlayers)

    return T    


@composite
def pressures(draw, min_layers=2, max_layers=30):

    nlayers = draw(integers(min_value=min_layers, max_value=max_layers))
    min_P = draw(integers(min_value=4, max_value=6))
    max_P = draw(integers(min_value=-6, max_value=3))

    P = np.logspace(min_P, max_P, nlayers)

    return P


@composite
def TPs(draw, min_layers=2, max_layers=30):
    nlayers = draw(integers(min_value=min_layers, max_value=max_layers))

    min_T = draw(floats(min_value=100.0, max_value=3000.0, allow_nan=False))
    max_T = draw(floats(min_value=100.0, max_value=3000.0, allow_nan=False))
    T = np.linspace(min_T, max_T, nlayers)

    min_P = draw(integers(min_value=4, max_value=6))
    max_P = draw(integers(min_value=-6, max_value=3))

    P = np.logspace(min_P, max_P, nlayers)

    return T, P, nlayers


@composite
def TP_npoints(draw, min_layers=2, max_layers=30):

    P = draw(pressures(min_layers=min_layers, max_layers=max_layers))
    nlayers = P.shape[0]

    T_top = draw(floats(min_value=100.0, max_value=3000.0, allow_nan=False))
    T_surface = draw(floats(min_value=100.0, max_value=3000.0, allow_nan=False))

    P_top = draw(integers(min_value=-6, max_value=6))
    P_top = 10**P_top
    P_surface = -1

    leftover = nlayers - 2

    if leftover > 0:

        temp_points = draw(lists(floats(min_value=100.0, max_value=3000.0,allow_nan=False), min_size=leftover, max_size=leftover))
        press_points = draw(lists(floats(min_value=1e-6, max_value=1e6,allow_nan=False), min_size=leftover, max_size=leftover))
    else:
        temp_points = []
        press_points = []

    return nlayers, T_top, T_surface, P_top, P_surface, temp_points,\
        press_points, P


@composite
def planets(draw, mass_range=[0.001, 10.0],
            radius_range=[0.001, 10.0]):
    from taurex.planet import Planet
    planet_mass = draw(floats(min_value=mass_range[0],
                              max_value=mass_range[1],
                              allow_nan=False))
    planet_radius = draw(floats(min_value=radius_range[0],
                                max_value=radius_range[1],
                                allow_nan=False))
    planet_distance = draw(floats(allow_nan=False))
    impact_param = draw(floats(allow_nan=False))
    orbital_period = draw(floats(allow_nan=False))
    albedo = draw(floats(allow_nan=False))
    transit_time = draw(floats(allow_nan=False))

    return Planet(planet_mass=planet_mass,
                  planet_radius=planet_radius,
                  planet_distance=planet_distance,
                  impact_param=impact_param,
                  orbital_period=orbital_period,
                  albedo=albedo,
                  transit_time=transit_time)