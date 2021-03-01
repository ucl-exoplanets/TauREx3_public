from .contribution import Contribution
import numpy as np
from taurex.data.fittable import fitparam


class LeeMieContribution(Contribution):
    """
    Computes Mie scattering contribution to optica depth
    Formalism taken from: Lee et al. 2013, ApJ, 778, 97

    Parameters
    ----------

    lee_mie_radius: float
        Particle radius in um

    lee_mie_q: float
        Extinction coefficient

    lee_mie_mix_ratio: float
        Mixing ratio in atmosphere

    lee_mie_bottomP: float
        Bottom of cloud deck in Pa

    lee_mie_topP: float
        Top of cloud deck in Pa


    """

    def __init__(self, lee_mie_radius=0.01, lee_mie_q=40,
                 lee_mie_mix_ratio=1e-10, lee_mie_bottomP=-1,
                 lee_mie_topP=-1):
        super().__init__('Mie')

        self._mie_radius = lee_mie_radius
        self._mie_q = lee_mie_q
        self._mie_mix = lee_mie_mix_ratio
        self._mie_bottom_pressure = lee_mie_bottomP
        self._mie_top_pressure = lee_mie_topP

    @fitparam(param_name='lee_mie_radius',
              param_latex='$R^{lee}_{\mathrm{mie}}$',
              default_fit=False,
              default_bounds=[0.01, 0.5])
    def mieRadius(self):
        """
        Particle radius in um
        """
        return self._mie_radius

    @mieRadius.setter
    def mieRadius(self, value):
        self._mie_radius = value

    @fitparam(param_name='lee_mie_q', param_latex='$Q_\mathrm{ext}$',
              default_fit=False, default_bounds=[-10, 1])
    def mieQ(self):
        """
        Extinction coefficient
        """
        return self._mie_q

    @mieQ.setter
    def mieQ(self, value):
        self._mie_q = value

    @fitparam(param_name='lee_mie_topP',
              param_latex='$P^{lee}_\mathrm{top}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[-1, 1])
    def mieTopPressure(self):
        """
        Pressure at top of cloud deck in Pa
        """
        return self._mie_top_pressure

    @mieTopPressure.setter
    def mieTopPressure(self, value):
        self._mie_top_pressure = value

    @fitparam(param_name='lee_mie_bottomP',
              param_latex='$P^{lee}_\mathrm{bottom}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[-1, 1])
    def mieBottomPressure(self):
        """
        Pressure at bottom of cloud deck in Pa
        """
        return self._mie_bottom_pressure

    @mieBottomPressure.setter
    def mieBottomPressure(self, value):
        self._mie_bottom_pressure = value

    @fitparam(param_name='lee_mie_mix_ratio',
              param_latex='$\chi^{lee}_\mathrm{mie}$',
              default_mode='log',
              default_fit=False,
              default_bounds=[-1, 1])
    def mieMixing(self):
        """
        Mixing ratio in atmosphere
        """
        return self._mie_mix

    @mieMixing.setter
    def mieMixing(self, value):
        self._mie_mix = value

    def prepare_each(self, model, wngrid):
        """
        Computes and weights the mie opacity for
        the pressure regions given

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid

        Yields
        ------
        component: :obj:`tuple` of type (str, :obj:`array`)
            ``Lee`` and the weighted mie opacity.

        """
        self._nlayers = model.nLayers
        self._ngrid = wngrid.shape[0]

        pressure_profile = model.pressureProfile

        bottom_pressure = self.mieBottomPressure
        if bottom_pressure < 0:
            bottom_pressure = pressure_profile[0]

        top_pressure = self.mieTopPressure
        if top_pressure < 0:
            top_pressure = pressure_profile[-1]

        wltmp = 10000/wngrid

        a = self.mieRadius

        x = 2.0 * np.pi * a / wltmp
        self.debug('wngrid %s', wngrid)
        self.debug('x %s', x)
        Qext = 5.0 / (self.mieQ * x**(-4.0) + x**(0.2))

        sigma_xsec = np.zeros(shape=(self._nlayers, wngrid.shape[0]))

        # This must transform um to the xsec format in TauREx (m2)
        am = a * 1e-6

        sigma_mie = Qext * np.pi * (am**2.0)

        self.debug('Qext %s', Qext)
        self.debug('radius um %s', a)
        self.debug('sigma %s', sigma_mie)

        self.debug('bottome_pressure %s', bottom_pressure)
        self.debug('top_pressure %s', top_pressure)

        cloud_filter = (pressure_profile <= bottom_pressure) & \
            (pressure_profile >= top_pressure)

        sigma_xsec[cloud_filter, ...] = sigma_mie * self.mieMixing

        self.sigma_xsec = sigma_xsec

        self.debug('final xsec %s', self.sigma_xsec)

        yield 'Lee', sigma_xsec

    def write(self, output):
        contrib = super().write(output)
        contrib.write_scalar('lee_mie_radius', self._mie_radius)
        contrib.write_scalar('lee_mie_q', self._mie_q)
        contrib.write_scalar('lee_mie_mix_ratio', self._mie_mix)
        contrib.write_scalar('lee_mie_bottomP', self._mie_bottom_pressure)
        contrib.write_scalar('lee_mie_topP', self._mie_top_pressure)
        return contrib

    @classmethod
    def input_keywords(self):
        return ['LeeMie', ]
    
    BIBTEX_ENTRIES = [
        """
        @article{Lee_2013,
            doi = {10.1088/0004-637x/778/2/97},
            url = {https://doi.org/10.1088%2F0004-637x%2F778%2F2%2F97},
            year = 2013,
            month = {nov},
            publisher = {{IOP} Publishing},
            volume = {778},
            number = {2},
            pages = {97},
            author = {Jae-Min Lee and Kevin Heng and Patrick G. J. Irwin},
            title = {{ATMOSPHERIC} {RETRIEVAL} {ANALYSIS} {OF} {THE} {DIRECTLY} {IMAGED} {EXOPLANET} {HR} 8799b},
            journal = {The Astrophysical Journal},
            abstract = {Directly imaged exoplanets are unexplored laboratories for the application of the spectral and temperature retrieval method, where the chemistry and composition of their atmospheres are inferred from inverse modeling of the available data. As a pilot study, we focus on the extrasolar gas giant HR 8799b, for which more than 50 data points are available. We upgrade our non-linear optimal estimation retrieval method to include a phenomenological model of clouds that requires the cloud optical depth and monodisperse particle size to be specified. Previous studies have focused on forward models with assumed values of the exoplanetary properties; there is no consensus on the best-fit values of the radius, mass, surface gravity, and effective temperature of HR 8799b. We show that cloud-free models produce reasonable fits to the data if the atmosphere is of super-solar metallicity and non-solar elemental abundances. Intermediate cloudy models with moderate values of the cloud optical depth and micron-sized particles provide an equally reasonable fit to the data and require a lower mean molecular weight. We report our best-fit values for the radius, mass, surface gravity, and effective temperature of HR 8799b. The mean molecular weight is about 3.8, while the carbon-to-oxygen ratio is about unity due to the prevalence of carbon monoxide. Our study emphasizes the need for robust claims about the nature of an exoplanetary atmosphere to be based on analyses involving both photometry and spectroscopy and inferred from beyond a few photometric data points, such as are typically reported for hot Jupiters.}
        }
        """,
    ]