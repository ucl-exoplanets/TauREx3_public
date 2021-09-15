"""The main taurex program"""


def parse_keywords(keywords):
    import tabulate
    from taurex.parameter.classfactory import ClassFactory
    cf = ClassFactory()
    print('\n')
    if keywords in ('contribs', ):
        print('')
        print('-----------------------------------------------')
        print('-------------Available Contributions-----------')
        print('-----------------------------------------------')
        print('')
        table = [(f'[[{c.__name__}]]',
                  c.__module__.split('.')[0].split('_')[-1])
                 for c in cf.contributionKlasses
                 if hasattr(c, 'input_keywords')]
        print(tabulate.tabulate(table,
                                headers=['Header', 'Source'],
                                tablefmt="fancy_grid"))

    elif keywords in ('chemistry', ):
        print('')
        print('-----------------------------------------------')
        print('-------------Available [Chemistry]-------------')
        print('-----------------------------------------------')
        print('')
        table = [(' / '.join(c.input_keywords()),
                  f'{c.__name__}',
                  c.__module__.split('.')[0].split('_')[-1])
                 for c in cf.chemistryKlasses if hasattr(c, 'input_keywords')]
        print(tabulate.tabulate(table,
              headers=['chemistry_type', 'Class', 'Source'],
              tablefmt="fancy_grid"))
        print('\n')
    elif keywords in ('temperature', ):
        print('')
        print('-----------------------------------------------')
        print('-------------Available [Temperature]-----------')
        print('-----------------------------------------------')
        print('')
        table = [(' / '.join(c.input_keywords()),
                  f'{c.__name__}',
                  c.__module__.split('.')[0].split('_')[-1])
                 for c in cf.temperatureKlasses
                 if hasattr(c, 'input_keywords')]
        print(tabulate.tabulate(table, 
                                headers=['profile_type', 'Class', 'Source'],
                                tablefmt="fancy_grid"))
        print('\n')
    elif keywords in ('gas', ):
        print('')
        print('-----------------------------------------------')
        print('-------------Available Gas Profiles------------')
        print('-----------------------------------------------')
        print('')
        table = [(' / '.join(c.input_keywords()),
                  f'{c.__name__}',
                  c.__module__.split('.')[0].split('_')[-1])
                 for c in cf.gasKlasses if hasattr(c, 'input_keywords')]

        print(tabulate.tabulate(table, headers=['gas_type', 'Class', 'Source'],
                                tablefmt="fancy_grid"))
        print('\n')
    elif keywords in ('optimizer', ):
        print('')
        print('-----------------------------------------------')
        print('-------------Available Optimizers--------------')
        print('-----------------------------------------------')
        print('')
        table = [(' / '.join(c.input_keywords()),
                 f'{c.__name__}',
                  c.__module__.split('.')[0].split('_')[-1])
                 for c in cf.optimizerKlasses if hasattr(c, 'input_keywords')]
        print(tabulate.tabulate(table, headers=['optimizer',
                                                'Class',
                                                'Source'],
                                tablefmt="fancy_grid"))
        print('\n')
    elif keywords in ('prior', ):
        print('')
        print('-----------------------------------------------')
        print('-------------Available Priors------------------')
        print('-----------------------------------------------')
        print('')
        table = [(f'{c.__name__}',
                 c.__module__.split('.')[0].split('_')[-1])
                 for c in cf.priorKlasses
                 if hasattr(c, 'input_keywords')]
        print(tabulate.tabulate(table, 
                                headers=['prior', 'Class', 'Source'],
                                tablefmt="fancy_grid"))
        print('\n')
    elif keywords in ('model', ):
        print('')
        print('-----------------------------------------------')
        print('-------------Available Forward [Model]s--------')
        print('-----------------------------------------------')
        print('')
        table = [(' / '.join(c.input_keywords()),
                  f'{c.__name__}', c.__module__.split('.')[0].split('_')[-1])
                 for c in cf.modelKlasses if hasattr(c, 'input_keywords')]
        print(tabulate.tabulate(table,
                                headers=['model_type', 'Class', 'Source'],
                                tablefmt="fancy_grid"))
        print('\n')
    elif keywords in ('pressure', ):
        print('')
        print('-----------------------------------------------')
        print('-------------Available [Pressure]s-------------')
        print('-----------------------------------------------')
        print('')
        table = [(' / '.join(c.input_keywords()), f'{c.__name__}',
                 c.__module__.split('.')[0].split('_')[-1])
                 for c in cf.pressureKlasses]
        print(tabulate.tabulate(table,
                                headers=['profile_type', 'Class', 'Source'], 
                                tablefmt="fancy_grid"))
        print('\n')


def show_parameters(model):
    import tabulate
    print('')
    print('-----------------------------------------------')
    print('------Available Retrieval Parameters-----------')
    print('-----------------------------------------------')
    print('')

    keywords = [k for k, v in model.fittingParameters.items()]

    short_desc = []
    for k, v in model.fittingParameters.items():
        doc = v[2].__doc__
        if doc is None or doc == 'None':
            short_desc.append('')
        else:
            split = doc.split('\n')
            for spl in split:
                if len(spl) > 0:
                    s = spl
                    break

            short_desc.append(s)

    output = tabulate.tabulate(zip(keywords,  short_desc),
                               headers=['Param Name', 'Short Desc'],
                               tablefmt="fancy_grid")
    print(output)
    print('\n\n')

    import tabulate
    print('')
    print('-----------------------------------------------')
    print('------Available Computable Parameters----------')
    print('-----------------------------------------------')
    print('')

    keywords = [k for k, v in model.derivedParameters.items()]

    short_desc = []
    for k, v in model.derivedParameters.items():
        doc = v[2].__doc__
        if doc is None or doc == 'None':
            short_desc.append('')
        else:
            split = doc.split('\n')
            for spl in split:
                if len(spl) > 0:
                    s = spl
                    break

            short_desc.append(s)

    output = tabulate.tabulate(zip(keywords,  short_desc),
                               headers=['Param Name', 'Short Desc'],
                               tablefmt="fancy_grid")
    print(output)
    print('\n\n')


def show_plugins():
    from taurex.parameter.classfactory import ClassFactory
    from taurex.log import setLogLevel
    import logging
    setLogLevel(logging.ERROR)

    successful_plugins, failed_plugins = ClassFactory().discover_plugins()

    print('\nSuccessfully loaded plugins')
    print('---------------------------')
    for k, v in successful_plugins.items():
        print(k)

    print('\n\nFailed plugins')
    print('---------------------------')
    for k, v in failed_plugins.items():
        print(k)
        print(f'Reason: {v}')
    
    print('\n')


def output_citations(model, instrument, optimizer):
    from taurex.mpi import barrier, get_rank

    barrier()
    bib_tex = None
    citation_string = None
    if get_rank() == 0:
        print('\n\n----------------------------------------------------------')
        print('----------------------Bibiliography-----------------------')
        print('----------------------------------------------------------')

        print('If you use any of the results from this run please cite')
        print('the following publications:')
        
        citation_string = ''
        all_citations = []
        print('\n')
        print('TauREx-Related')
        print('--------------\n')
        from taurex._citation import __citations__, taurex_citation
        citation_string += __citations__
        all_citations.extend(taurex_citation.citations())
        print(__citations__)

        print('Forward model')
        print('-------------\n')

        cite = model.nice_citation()
        all_citations.extend(model.citations())
        citation_string += cite
        print(cite)

        if optimizer is not None:
            cite = optimizer.nice_citation()
            all_citations.extend(optimizer.citations())
            if len(cite) > 0:
                citation_string += cite
                print('Optimizer')
                print('---------\n')
                print(cite)

        if instrument is not None:
            cite = instrument.nice_citation()
            all_citations.extend(instrument.citations())
            if len(cite) > 0:
                citation_string += cite
                print('Instrument')
                print('---------\n')
                print(cite)

        from taurex.core import to_bibtex
        bib_tex = to_bibtex(all_citations)
    
    barrier()

    return bib_tex, citation_string

def only_bibtex(filename, pp):
    model = pp.generate_appropriate_model()
    instrument = pp.generate_instrument()[0]
    optimizer = pp.generate_optimizer()

    bib_tex, citation_string = output_citations(model, instrument, optimizer)
    if bib_tex:
        with open(filename, 'w') as f:
            f.write(bib_tex)



def main():
    import argparse
    import datetime

    import logging
    from taurex.mpi import get_rank
    from taurex.log import setLogLevel
    from taurex.log.logger import root_logger
    from taurex.parameter import ParameterParser
    from taurex.output.hdf5 import HDF5Output
    from taurex.util.output import store_contributions
    from .taurexdefs import OutputSize
    from . import __version__ as version

    import numpy as np

    parser = argparse.ArgumentParser(description='TauREx {}'.format(version))

    parser.add_argument("-i", "--input", dest='input_file', type=str,
                        help="Input par file to pass")

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

    parser.add_argument('-v', "--version", dest='version', default=False,
                        help="Display version", action='store_true')

    parser.add_argument("--plugins", dest='plugins', default=False,
                        help="Display plugins", action='store_true')

    parser.add_argument("--fitparams", dest='fitparams', default=False,
                        help="Display available fitting params", 
                        action='store_true')

    parser.add_argument("--bibtex", dest='bibtex', type=str,
                        help="Output bibliography .bib to filepath")

    parser.add_argument("--only-bib", dest='no_run',
                        help="Do not run anything, only store bibtex (must have --bibtex)", default=False, action='store_true')

    parser.add_argument("--keywords", dest="keywords", type=str)

    args = parser.parse_args()

    output_size = OutputSize.heavy

    if args.version:
        print(version)
        return

    if args.plugins:
        show_plugins()
        return

    if args.keywords:
        parse_keywords(args.keywords)
        return

    if args.input_file is None:
        print('Fatal: No input file specified.')
        return

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

    if args.no_run and args.bibtex:
        return only_bibtex(args.bibtex, pp)

    # Get the spectrum
    observation = pp.generate_observation()

    binning = pp.generate_binning()


    # Generate a model from the input
    model = pp.generate_appropriate_model(obs=observation)

    # build the model
    model.build()

    if args.fitparams:
        show_parameters(model)
        return



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
        pp.setup_optimizer(optimizer)

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

            profiles = model.generate_profiles()
            spectrum = \
                binning.generate_spectrum_output(result,
                                                 output_size=output_size)

            if inst_result is not None:
                spectrum['instrument_wngrid'] = inst_result[0]
                spectrum['instrument_wnwidth'] = inst_result[-1]
                spectrum['instrument_wlgrid'] = 10000/inst_result[0]
                spectrum['instrument_spectrum'] = inst_result[1]
                spectrum['instrument_noise'] = inst_result[2]

            try:
                spectrum['Contributions'] = \
                    store_contributions(binning, model, 
                                        output_size=output_size-3)
            except Exception:
                pass

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

    bib_tex, citation_string = output_citations(model, instrument, optimizer)

    if args.output_file and bib_tex and citation_string:
        with HDF5Output(args.output_file, append=True) as o:
            bib = o.create_group('Bibliography')
            bib.write_string('short_form', citation_string)
            bib.write_string('bibtex', bib_tex)

    if args.bibtex and bib_tex and citation_string:
        with open(args.bibtex, 'w') as f:
            f.write(bib_tex)

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
