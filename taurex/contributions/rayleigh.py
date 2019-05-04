 
from .contribution import Contribution
import numpy as np
import numba

@numba.jit(nopython=True,parallel=True, nogil=True)
def rayleigh_numba(sigma,density,path,nlayers,ngrid,nmols,layer):
    tau = np.zeros(shape=(ngrid,))
    for k in range(nlayers-layer):
        _path = path[k]
        _density = density[k+layer]
        for mol in range(nmols):
            for wn in range(ngrid):
                tau[wn] += sigma[k+layer,mol,wn]*_path*_density
    return tau


class RayleighContribution(Contribution):


    def __init__(self):
        super().__init__('Rayleigh')
 

    def contribute(self,model,layer,density,path_length,return_contrib):
        import numexpr as ne
        total_layers = model.pressure_profile.nLayers
        # sigma = self.sigma_rayleigh[layer:total_layers,:]
        # #print(sigma.shape,density.shape,path_length.shape)
        # combined_pt_dt = (density*path_length)[:,None,None]
        # contrib = ne.evaluate('sum(sigma*combined_pt_dt,axis=0)')

        # contrib = ne.evaluate('sum(contrib,axis=0)')
        #contrib = np.sum(sigma*density*path_length,axis=0)
        #contrib = np.sum(contrib,axis=0)
        contrib = rayleigh_numba(self.sigma_rayleigh,density,path_length,self._nlayers,self._ngrid,self._nmols,layer)
        if return_contrib:
            self._total_contrib[layer,:]+=contrib
        return contrib

    def build(self,model):
        pass
    

    def finalize(self,model):
        raise NotImplementedError

    @property
    def totalContribution(self):
        return self._total_contrib
 
    def prepare(self,model,wngrid):

        # precalculate rayleigh scattering cross sections

        self.info('Compute Rayleigh scattering cross sections')

        sigma_rayleigh_dict = {}

        molecules = model._gas_profile.active_gases + model._gas_profile.inactive_gases

        for gasname in molecules:

            gasname = gasname.upper()

            # get the refractive index. Formulae taken from Allen Astrophysical Quantities if not otherwise specified
            n_formula = True # assume we have a formula for the refractive index of gasname
            king = 1 # King correction factor
            ns = 0   # refractive index
            wn = wngrid # wavenumber in cm^-1
            wno = ((10000. / wn) * 1.e-6)

            with np.errstate(divide='ignore'):
                wltmp = 10000./wn
            if gasname == 'HE':
                 # C. R. Mansfield and E. R. Peck. Dispersion of helium, J. Opt. Soc. Am. 59, 199-203 (1969)
                ns = 1 + 0.01470091/(423.98-(wltmp)**-2)
                # king is one for He
            elif gasname == 'H2':
                ns = 1 + 13.58e-5 * (1. + 7.52e-3 / (wltmp)**2)
                delta = 0.035 # from Morgan old code..
                king = (6.+3.*delta)/(6.-7.*delta) # Bates (1984)
            elif gasname == 'N2':
                ns = 1. + (6498.2 + (307.43305e12)/(14.4e9 - wn**2))*1.e-8 # Peck and Khanna
                king = 1.034+3.17e-12*wn**2 # Bates
            elif gasname == 'O2':
                #  J. Zhang, Z. H. Lu, and L. J. Wang. Appl. Opt. 47, 3143-3151 (2008)
                ns = 1 + 1.181494e-4 + 9.708931e-3/(75.4-(wltmp)**-2)
                king = 1.096
            elif gasname == 'CO2':
                #A. Bideau-Mehu, Y. Guern, R. Abjean and A. Johannin-Gilles. Opt. Commun. 9, 432-434 (1973)
                ns = 1 + 6.991e-2/(166.175-(wltmp)**-2)+1.44720e-3/(79.609-(wltmp)**-2)+6.42941e-5/(56.3064-(wltmp)**-2)\
                     +5.21306e-5/(46.0196-(wltmp)**-2)+1.46847e-6/(0.0584738-(wltmp)**-2)
                king = 1.1364 #  Sneep & Ubachs 2005
            elif gasname == 'CH4':
                ns = 1 + 1.e-8*(46662. + 4.02*1.e-6*(1/((wltmp)*1.e-4))**2)
            elif gasname == 'CO':
                ns = 1 + 32.7e-5 * (1. + 8.1e-3 / (wltmp)**2)
                king = 1.016  #  Sneep & Ubachs 2005
            elif gasname == 'NH3':
                ns = 1 + 37.0e-5 * (1. + 12.0e-3 / (wltmp)**2)
            elif gasname == 'H2O':
                # P. E. Ciddor. Appl. Optics 35, 1566-1573 (1996)
                ns_air = (1 + (0.05792105/(238.0185 - (wltmp)**-2) + 0.00167917/(57.362-(wltmp)**-2)))
                ns = 0.85 * (ns_air - 1.) + 1  # ns = 0.85 r(air) (Edlen 1966)
                delta = 0.17 # Marshall & Smith 1990
                king = (6.+3.*delta)/(6.-7.*delta)
            else:
                # this sets sigma_R to zero for all other gases
                n_formula = False
                self.warning('There is no formula for the refractive index of %s. '
                                'Cannot compute the cross section' % gasname)


            if n_formula: # only if the refractive index was computed
                Ns = 2.6867805e25 # in m^-3
                with np.errstate(divide='ignore'):
                    sigma = (24.*np.pi**3)/(Ns**2) *(((ns**2-1.)/(ns**2+2.))**2) * king/(wno**4)

                # override H2 and He sigma with formulae from M Line
                with np.errstate(divide='ignore'):
                    wave = (1/wn)*1E8
                if gasname == 'H2':
                    sigma = ((8.14E-13)*(wave**(-4.))*(1+(1.572E6)*(wave**(-2.))+(1.981E12)*(wave**(-4.))))*1E-4
                if gasname == 'HE':
                    sigma =  ((5.484E-14)*(wave**(-4.))*(1+(2.44E5)*(wave**(-2.))))*1E-4
                    #sigma[:] = 0

                self.info('Rayleigh scattering cross section of %s correctly computed' % (gasname))
                sigma_rayleigh_dict[gasname] = sigma
            #else:
            #    sigma_rayleigh_dict[gasname] = np.zeros((len(wn)))

        self.sigma_rayleigh_dict = sigma_rayleigh_dict

        self._ngrid = wngrid.shape[0]
        self._nmols = len(self.sigma_rayleigh_dict.keys())
        self._nlayers = model.pressure_profile.nLayers

        self.sigma_rayleigh = np.zeros(shape=(model.pressure_profile.nLayers,self._nmols,wngrid.shape[0]))
        self.info('Computing Ray interpolation ')
        for rayleigh_idx,rayleigh in enumerate(self.sigma_rayleigh_dict.items()):
            gas,xsec = rayleigh
            for idx_layer in range(model.nLayers):
                
               
                ray_factor = model._gas_profile.get_gas_mix_profile(gas)
                

                self.sigma_rayleigh[idx_layer,rayleigh_idx]= ray_factor[idx_layer]*xsec[:]
        self.info('DONE!!!')
        self._total_contrib = np.zeros(shape=(model.pressure_profile.nLayers,wngrid.shape[0],))