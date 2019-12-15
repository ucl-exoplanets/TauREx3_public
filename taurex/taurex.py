"""The main taurex program"""


def main():
    import argparse
    import datetime

    import logging
    from taurex.mpi import get_rank
    from taurex.log import setLogLevel
    from taurex.log.logger import root_logger
    from taurex.parameter import ParameterParser
    from taurex.output.hdf5 import HDF5Output
    from taurex.util.output import generate_profile_dict, store_contributions
    from .taurexdefs import OutputSize
    from . import __version__ as version

    import numpy as np

    parser = argparse.ArgumentParser(description='TauREx {}'.format(version))

    parser.add_argument("-i", "--input", dest='input_file', type=str,
                        required=True, help="Input par file to pass")

    parser.add_argument("-R", "--retrieval", dest='retrieval', default=False,
                        help="When set, runs retrieval", action='store_true')

    parser.add_argument("-p", "--plot", dest='plot', default=False,
                              help="Whether to plot after the run",
                              action='store_true')

    parser.add_argument("-g", "--debug-log", dest='debug', default=False,
                        help="Debug log output", action='store_true')

    parser.add_argument("-c", "--show-contrib", dest='contrib',
                        default=False, help="Show basic contributions",
                        action='store_true')

    parser.add_argument("-C", "--full-contrib", dest='full_contrib',
                        default=False, help="Show ALL contributions",
                        action='store_true')

    parser.add_argument("--light", dest='light', default=False,
                        help="Light outputs", action='store_true')

    parser.add_argument("--lighter", dest='lighter', default=False,
                        help="Even Lighter outputs", action='store_true')

    parser.add_argument("-o", "--output_file", dest='output_file', type=str)

    parser.add_argument("-S", "--save-spectrum",
                        dest='save_spectrum', type=str)
    args = parser.parse_args()

    output_size = OutputSize.heavy

    if args.light:
        output_size = OutputSize.light

    if args.lighter:
        output_size = OutputSize.lighter

    if args.debug:
        setLogLevel(logging.DEBUG)

    root_logger.info('TauREx %s', version)

    root_logger.info('TauREx PROGRAM START AT %s', datetime.datetime.now())

    # Parse the input file
    pp = ParameterParser()
    pp.read(args.input_file)

    # Setup global parameters
    pp.setup_globals()
    # Generate a model from the input
    model = pp.generate_appropriate_model()

    # build the model
    model.build()

    # Get the spectrum
    observation = pp.generate_observation()

    binning = pp.generate_binning()

    wngrid = None

    if binning == 'observed' and observation is None:
        logging.critical('Binning selected from Observation yet None provided')
        quit()

    if binning is None:
        if observation is None or observation == 'self':
            binning = model.defaultBinner()
            wngrid = model.nativeWavenumberGrid
        else:
            binning = observation.create_binner()
            wngrid = observation.wavenumberGrid
    else:
        if binning == 'native':
            binning = model.defaultBinner()
            wngrid = model.nativeWavenumberGrid
        elif binning == 'observed':
            binning = observation.create_binner()
            wngrid = observation.wavenumberGrid
        else:
            binning, wngrid = binning

    instrument = pp.generate_instrument(binner=binning)

    num_obs = 1
    if instrument is not None:
        instrument, num_obs = instrument

    if observation == 'self' and instrument is None:
        logging.getLogger('taurex').critical(
            'Instrument nust be specified when using self option')
        raise ValueError('No instruemnt specified for self option')

    inst_result = None
    if instrument is not None:
        inst_result = instrument.model_noise(
            model, model_res=model.model(), num_observations=num_obs)

    # Observation on self
    if observation == 'self':
        from taurex.data.spectrum import ArraySpectrum
        from taurex.util.util import wnwidth_to_wlwidth
        inst_wngrid, inst_spectrum, inst_noise, inst_width = inst_result

        inst_wlgrid = 10000/inst_wngrid

        inst_wlwidth = wnwidth_to_wlwidth(inst_wngrid, inst_width)
        observation = ArraySpectrum(
            np.vstack([inst_wlgrid, inst_spectrum,
                       inst_noise, inst_wlwidth]).T)
        binning = observation.create_binner()

    # Handle outputs
    if args.output_file:
        # Output taurex data
        with HDF5Output(args.output_file) as o:
            model.write(o)

    optimizer = None
    solution = None

    if args.retrieval is True:
        import time
        if observation is None:
            logging.critical('No spectrum is defined!!')
            quit()

        optimizer = pp.generate_optimizer()
        optimizer.set_model(model)
        optimizer.set_observed(observation)

        fitting_parameters = pp.generate_fitting_parameters()

        for key, value in fitting_parameters.items():
            fit = value['fit']
            bounds = value['bounds']
            mode = value['mode']
            factor = value['factor']

            if fit:
                logging.info('Fitting: {}'.format(key))
                optimizer.enable_fit(key)
            else:
                optimizer.disable_fit(key)

            if factor:
                optimizer.set_factor_boundary(key, factor)

            if bounds:
                optimizer.set_boundary(key, bounds)

            if mode:
                optimizer.set_mode(key, mode.lower())

        start_time = time.time()
        solution = optimizer.fit(output_size=output_size)

        end_time = time.time()

        root_logger.info(
            'Total Retrieval finish in %s seconds', end_time-start_time)

        for _, optimized, _, _ in optimizer.get_solution():
            optimizer.update_model(optimized)
            break

    result = model.model()

    if args.save_spectrum is not None:

        # with open(args.save_spectrum, 'w') as f:
        from taurex.util.util import wnwidth_to_wlwidth, compute_bin_edges
        save_wnwidth = compute_bin_edges(wngrid)[1]
        save_wl = 10000/wngrid
        save_wlwidth = wnwidth_to_wlwidth(wngrid, save_wnwidth)
        save_model = binning.bin_model(result)[1]
        save_error = np.zeros_like(save_wl)
        if inst_result is not None:
            inst_wngrid, inst_spectrum, inst_noise, inst_width = inst_result

            save_model = inst_spectrum
            save_wl = 10000/inst_wngrid

            save_wlwidth = wnwidth_to_wlwidth(inst_wngrid, inst_width)

            save_error = inst_noise

        np.savetxt(args.save_spectrum,
                   np.vstack((save_wl, save_model, save_error,
                              save_wlwidth)).T)

    if args.output_file:

        # Output taurex data
        with HDF5Output(args.output_file, append=True) as o:

            out = o.create_group('Output')
            if observation is not None:
                obs = o.create_group('Observed')
                observation.write(obs)

            profiles = generate_profile_dict(model)
            spectrum = \
                binning.generate_spectrum_output(result,
                                                 output_size=output_size)

            if inst_result is not None:
                spectrum['instrument_wngrid'] = inst_result[0]
                spectrum['instrument_wnwidth'] = inst_result[-1]
                spectrum['instrument_wlgrid'] = 10000/inst_result[0]
                spectrum['instrument_spectrum'] = inst_result[1]
                spectrum['instrument_noise'] = inst_result[2]

            spectrum['Contributions'] = \
                store_contributions(binning, model, output_size=output_size-3)
            if solution is not None:
                out.store_dictionary(solution, group_name='Solutions')
                priors = {}
                priors['Profiles'] = profiles
                priors['Spectra'] = spectrum
                out.store_dictionary(priors, group_name='Priors')
            else:
                out.store_dictionary(profiles, group_name='Profiles')
                out.store_dictionary(spectrum, group_name='Spectra')

            if optimizer:
                optimizer.write(o)

    root_logger.info('TauREx PROGRAM END AT %s s', datetime.datetime.now())

    if args.plot:
        wlgrid = 10000/wngrid
        if get_rank() == 0:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.set_xlabel(r'Wavelength $(\mu m)$')
            ax.set_ylabel(r'$(R_p/R_s)^2$')
            is_lightcurve = False
            try:
                from taurex.model.lightcurve.lightcurve import LightCurveModel
                is_lightcurve = isinstance(model, LightCurveModel)
                ax.set_xscale('linear')
            except ImportError:
                pass

            if observation is not None:
                if is_lightcurve:
                    ax.plot(observation.spectrum.flatten(),
                            label='observation')
                else:
                    ax.errorbar(observation.wavelengthGrid,
                                observation.spectrum, observation.errorBar,
                                fmt='.',
                                label='observation')

            if is_lightcurve:
                ax.plot(result[1], label='forward model')
            else:

                if inst_result is not None:
                    from taurex.util.util import wnwidth_to_wlwidth
                    inst_wngrid, inst_spectrum, \
                        inst_noise, inst_width = inst_result

                    inst_wlgrid = 10000/inst_wngrid

                    inst_wlwidth = wnwidth_to_wlwidth(inst_wngrid, inst_width)

                    ax.errorbar(inst_wlgrid, inst_spectrum, inst_noise,
                                inst_wlwidth/2, '.', label='Instrument')

                else:
                    ax.plot(wlgrid, binning.bin_model(
                        result)[1], label='forward model')

                ax.set_xscale('log')

            if args.contrib:
                native_grid, contrib_result = model.model_contrib(
                    wngrid=wngrid)

                for contrib_name, contrib in contrib_result.items():

                    flux, tau, extras = contrib

                    binned = binning.bindown(native_grid, flux)
                    if is_lightcurve:
                        ax.plot(binned[1], label=contrib_name)
                    else:
                        ax.plot(wlgrid, binned[1], label=contrib_name)

            if args.full_contrib:
                native_grid, contrib_result = model.model_full_contrib(
                    wngrid=wngrid)

                for contrib_name, contrib in contrib_result.items():

                    for name, flux, tau, extras in contrib:

                        label = '{} - {}'.format(contrib_name, name)

                        binned = binning.bindown(native_grid, flux)
                        if is_lightcurve:
                            ax.plot(binned[1], label=label)
                        else:
                            ax.plot(wlgrid, binned[1], label=label)

            plt.legend()
            plt.show()
        else:
            logging.getLogger('taurex').warning(
                'Number of processes > 1 so not plotting')


if __name__ == "__main__":

    main()
