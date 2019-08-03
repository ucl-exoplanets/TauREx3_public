"""The main taurex program"""


def write_spectrum(output,native_wngrid,model,tau,contributions,wngrid=None,binned_model=None,observed=None):
    spectrum = output.create_group('Spectrum')
    spectrum.write_array('native_wngrid',native_wngrid)
    spectrum.write_array('native_spectrum',model)
    spectrum.write_array('native_tau',tau)

    contrib = spectrum.create_group('native_contributions')
    for name,value in contributions:
        contrib.write_array(name,value)

    if wngrid is not None:
        spectrum.write_array('wngrid',wngrid)
    if binned_model is not None:
        spectrum.write_array('binned_spectrum',binned_model)
    if observed is not None:
        spectrum.write_array('observed_spectrum',observed.spectrum)
        spectrum.write_array('observed_wngrid',observed.wavenumberGrid)
        spectrum.write_array('observed_error',observed.errorBar)




def main():
    import argparse

    import os
    import logging
    import numpy as np
    from taurex.mpi import get_rank,nprocs
    from taurex.log import Logger,setLogLevel
    from taurex.parameter import ParameterParser
    from taurex.util import bindown
    from taurex.output.hdf5 import HDF5Output
    parser = argparse.ArgumentParser(description='Taurex')
    parser.add_argument("-i", "--input",dest='input_file',type=str,required=True,help="Input par file to pass")
    parser.add_argument("-R", "--retrieval",dest='retrieval',default=False, help="When set, runs retrieval",action='store_true')
    parser.add_argument("-p", "--plot",dest='plot',default=False,help="Whether to plot after the run",action='store_true')
    parser.add_argument("-g", "--debug-log",dest='debug',default=False,help="Debug log output",action='store_true')
    parser.add_argument("-c", "--show-contrib",dest='contrib',default=False,help="Show basic contributions",action='store_true')
    parser.add_argument("-C","--full-contrib",dest='full_contrib',default=False,help="Show ALL contributions",action='store_true')
    parser.add_argument("-o","--output_file",dest='output_file',type=str)


    args=parser.parse_args()

    if args.debug:
        setLogLevel(logging.DEBUG)
        #logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #Parse the input file
    pp = ParameterParser()
    pp.read(args.input_file)

    #Setup global parameters
    pp.setup_globals()
    #Generate a model from the input
    model = pp.generate_appropriate_model()

    #build the model
    model.build()

    #Get the spectrum
    observed,bindown_wngrid = pp.generate_spectrum()
    
    #logging.info('Using grid {}'.format(bindown_wngrid))
    #logging.info('Observed grid is {}'.format(observed.wavenumberGrid))
    #Get the native grid
    native_grid = model.nativeWavenumberGrid



    #If we bin down then get the appropriate grid
    if bindown_wngrid is None:
        bindown_wngrid = native_grid
    else:
        native_grid = native_grid[(native_grid >= bindown_wngrid.min()*0.9) & (native_grid<= bindown_wngrid.max()*1.1) ]

    optimizer = None

    if args.retrieval is True:
        if observed is None:
            logging.critical('No spectrum is defined!!')
            quit()
        
        optimizer = pp.generate_optimizer()
        optimizer.set_model(model)
        optimizer.set_observed(observed)

        fitting_parameters = pp.generate_fitting_parameters()

        for key,value in fitting_parameters.items():
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
                optimizer.set_factor_boundary(key,factor)

            if bounds:
                optimizer.set_boundary(key,bounds)
            
            if mode:
                optimizer.set_mode(key,mode.lower())

        logging.getLogger('taurex').setLevel(logging.WARNING)

        optimizer.fit()

        logging.getLogger('taurex').setLevel(logging.INFO)

    #Run the model

    new_absp,absp,tau,contrib=model.model(bindown_wngrid,return_contrib=True)
    #Get out new binned down model
    #new_absp = bindown(native_grid,absp,bindown_wngrid)

    if args.output_file and get_rank()==0:
        from taurex.util.output import store_taurex_results
        #Output taurex data
        with HDF5Output(args.output_file) as o:

            model.write(o)
            store_taurex_results(o,model,native_grid,absp,tau,contrib,observed=observed,optimizer=optimizer)

            ## old writing calls

            # write_spectrum(o,native_grid,absp,tau,contrib,bindown_wngrid,new_absp,observed)
            # if optimizer:
            #    optimizer.write(o)


            if args.plot:
                from taurex.util.output import plot_taurex_results_from_hdf5
                plot_taurex_results_from_hdf5(args.output_file)




    contrib_res = None

    if args.full_contrib:
        contrib_res = model.model_full_contrib(wngrid=bindown_wngrid)

    
    wlgrid = 10000/bindown_wngrid
    if args.plot:



        if get_rank()==0  and nprocs()<=1:
            import matplotlib.pyplot as plt
            fig= plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_xscale('log')
            ax.set_xlabel(r'Wavelength $(\mu m)$')
            ax.set_ylabel(r'$(R_p/R_s)^2$')
            is_lightcurve = False
            try:
                from taurex.model.lightcurve.lightcurve import LightCurveModel
                is_lightcurve = isinstance(model,LightCurveModel)
            except ImportError:
                pass

            if observed is not None:
                if is_lightcurve:
                    ax.plot(observed.spectrum.flatten(),label='observed')
                else:
                    ax.errorbar(observed.wavelengthGrid,observed.spectrum,observed.errorBar,label='observed')

            if is_lightcurve:
                ax.plot(new_absp,label='forward model')
            else:
            #Plot the absorption
                ax.plot(wlgrid,new_absp,label='forward model')


            if args.contrib and not is_lightcurve:
                for name,value in contrib:
                    new_value = bindown(native_grid,value,bindown_wngrid)
                    ax.plot(wlgrid,new_value,label='All {}'.format(name),alpha=0.8)
            if contrib_res is not None:
                for k,v in contrib_res.items():
                    first_name = k
                    for out in v:
                        second_name = out[0]
                        binned = out[1]
                        label='{}-{}'.format(first_name,second_name)
                        ax.plot(wlgrid,binned,label=label,alpha=0.6)

            
            #If we have an observation then plot it




            plt.legend()
            plt.show()
        
        else:
            logging.getLogger('taurex').warning('Number of processes > 1 so not plotting')
    



if __name__=="__main__":
    main()