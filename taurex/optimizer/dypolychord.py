from .polychord import PolyChordOptimizer
import numpy as np
import os
# Import some example python likelihoods
import dyPolyChord.python_likelihoods as likelihoods
import dyPolyChord.python_priors as priors  # Import some example python priors
import dyPolyChord.pypolychord_utils
import dyPolyChord
import time


class dyPolyChordOptimizer(PolyChordOptimizer):

    def __init__(self, polychord_path=None, observed=None, model=None,
                 num_live_points=1500,
                 max_iterations=0,
                 maximum_modes=100,
                 cluster=True,
                 evidence_tolerance=0.5,
                 mode_tolerance=-1e90,
                 resume=False,
                 verbosity=1, sigma_fraction=0.1):
        super().__init__(
            polychord_path=polychord_path, observed=observed, model=model,
            num_live_points=num_live_points,
            max_iterations=max_iterations,
            maximum_modes=maximum_modes,
            cluster=cluster,
            evidence_tolerance=evidence_tolerance,
            mode_tolerance=mode_tolerance,
            resume=resume,
            verbosity=verbosity, sigma_fraction=sigma_fraction)

    def compute_fit(self):
        self._polychord_output = None
        data = self._observed.spectrum
        datastd = self._observed.errorBar
        sqrtpi = np.sqrt(2*np.pi)

        ndim = len(self.fitting_parameters)

        def polychord_loglike(cube):
            # log-likelihood function called by polychord
            fit_params_container = np.array(
                [cube[i] for i in range(len(self.fitting_parameters))])
            chi_t = self.chisq_trans(fit_params_container, data, datastd)

            # print('---------START---------')
            # print('chi_t',chi_t)
            # print('LOG',loglike)
            loglike = -np.sum(np.log(datastd*sqrtpi)) - 0.5 * chi_t
            return loglike, [0.0]

        def polychord_uniform_prior(hypercube):
            # prior distributions called by polychord. Implements a uniform prior
            # converting parameters from normalised grid to uniform prior
            # print(type(cube))
            cube = [0.0]*ndim

            for idx, bounds in enumerate(self.fit_boundaries):
                # print(idx,self.fitting_parameters[idx])
                bound_min, bound_max = bounds
                cube[idx] = (hypercube[idx] *
                             (bound_max-bound_min)) + bound_min
                #print('CUBE idx',cube[idx])
            # print('-----------')
            return cube
        status = None

        datastd_mean = np.mean(datastd)

        likelihood = polychord_loglike
        prior = polychord_uniform_prior

        # Make a callable for running PolyChord
        my_callable = dyPolyChord.pypolychord_utils.RunPyPolyChord(
            likelihood, prior, ndim)

        # Specify sampler settings (see run_dynamic_ns.py documentation for more details)
        # whether to maximise parameter estimation or evidence accuracy.
        dynamic_goal = 1.0
        # number of live points to use in initial exploratory run.
        ninit = ndim*5
        # total computational budget is the same as standard nested sampling with nlive_const live points.
        nlive_const = ndim * 25
        settings_dict = {
            'num_repeats': ndim * 5,
            'do_clustering': self.do_clustering,
            'num_repeats': ndim,
            'precision_criterion': self.evidence_tolerance,
            'logzero': -1e70,
            'read_resume': self.resume,
            'base_dir': self.dir_polychord,
            'file_root': '1-'}

        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            # Run dyPolyChord
            dyPolyChord.run_dypolychord(my_callable, dynamic_goal, settings_dict,
                                        ninit=ninit, nlive_const=nlive_const, comm=comm)
        except ImportError:
            dyPolyChord.run_dypolychord(my_callable, dynamic_goal, settings_dict,
                                        ninit=ninit, nlive_const=nlive_const)

        time.sleep(2.0)

        #pypolychord.run_polychord(polychord_loglike, ndim, 1, settings, polychord_uniform_prior)
        self._polychord_output = self.store_polychord_solutions()

    @classmethod
    def input_keywords(self):
        return ['dypolychord', 'dynamic-polychord', ]
    
    BIBTEX_ENTRIES = [
        """
        @article{dypolychord2,
        author={Higson, Edward and Handley, Will and Hobson, Michael and Lasenby, Anthony},
        title={Dynamic nested sampling: an improved algorithm for parameter estimation and evidence calculation},
        year={2019},
        volume={29},
        number={5},
        pages={891--913},
        journal={Statistics and Computing},
        doi={10.1007/s11222-018-9844-0},
        url={https://doi.org/10.1007/s11222-018-9844-0},
        archivePrefix={arXiv},
        arxivId={1704.03459}}
        """,
        """
        @article{dypolychord1,
        title={dyPolyChord: dynamic nested sampling with PolyChord},
        author={Higson, Edward},
        year={2018},
        journal={Journal of Open Source Software},
        number={29},
        pages={916},
        volume={3},
        doi={10.21105/joss.00965},
        url={http://joss.theoj.org/papers/10.21105/joss.00965}}
        """

    ]
