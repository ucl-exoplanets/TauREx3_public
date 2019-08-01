import numpy as np

_molecule_func = {
        'He' : lambda x: compute_he(x),
        'H2' : lambda x: compute_h2(x),
        'N2' : lambda x: compute_ray_sigma(x,
                            n=1. + (6498.2 + (307.43305e12)/(14.4e9 - x**2))*1.e-8,
                            king=1.034+3.17e-12*x**2),
        'O2' : lambda x: compute_ray_sigma(x,
                            n=1 + 1.181494e-4 + 9.708931e-3/(75.4-(10000/x)**-2),
                            king = 1.096),
        'CO2' : lambda x: 
                (lambda wl_2=(10000./x)**-2 : 
                        compute_ray_sigma(x,
                        n=1 + 6.991e-2/(166.175-wl_2)+1.44720e-3/(79.609-wl_2)+6.42941e-5/(56.3064-wl_2)\
                                +5.21306e-5/(46.0196-wl_2)+1.46847e-6/(0.0584738-wl_2),
                        
                        king = 1.1364))(), 
        'CH4' : lambda x: compute_ray_sigma(x,
                                n=1 + 1.e-8*(46662. + 4.02*1.e-6*(1/((10000/x)*1.e-4))**2)),
        'CO' : lambda x: compute_ray_sigma(x,
                                n=1 + 32.7e-5 * (1. + 8.1e-3 / (10000/x)**2),king=1.016),
        'NH3' : lambda x: compute_ray_sigma(x,
                                n=1 + 37.0e-5 * (1. + 12.0e-3 / (10000/x)**2)),
        'H2O' : lambda x: 
                    (lambda ns_air=(1 + (0.05792105/(238.0185 - (10000/x)**-2) + 0.00167917/(57.362-(10000/x)**-2))),delta=0.17:
                        compute_ray_sigma(x,n=0.85 * (ns_air - 1.) + 1,king=(6.+3.*delta)/(6.-7.*delta)))(),
            


}



def rayleigh_sigma_from_name(molecule_name,wngrid):
    try:
        return _molecule_func[molecule_name](wngrid)
    except KeyError:
        return None


def compute_h2(wngrid):
    wave = 1e8/wngrid

    return ((8.14E-13)*(wave**(-4.))*(1+(1.572E6)*(wave**(-2.))+(1.981E12)*(wave**(-4.))))*1E-4

def compute_he(wngrid):
    wave = 1e8/wngrid

    return ((5.484E-14)*(wave**(-4.))*(1+(2.44E5)*(wave**(-2.))))*1E-4


def compute_ray_sigma(wngrid,n=0.0,n_air = 2.6867805e25,king=1.0):
    wlgrid = (10000.0/wngrid)*1e-6

    
    n_factor = (n**2 - 1)/(n_air*(n**2.0 + 2.0))
    sigma = 24.0*(np.pi**3.0)*king*(n_factor**2.0)/(wlgrid**4)

    return sigma