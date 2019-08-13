"""The main taurex program"""




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

    if args.output_file and get_rank()==0:
        from taurex.util.output import store_taurex_results,store_profiles,generate_profile_dict,generate_spectra_dict
        #Output taurex data
        with HDF5Output(args.output_file) as o:

            model.write(o)

    optimizer = None
    solution = None
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


        solution = optimizer.fit()

    #Run the model
    result=model.model(bindown_wngrid,return_contrib=True,cutoff_grid=False)
    new_absp,absp,tau,contrib = result
    #Get out new binned down model
    contrib_res = None

    if args.full_contrib or args.output_file:
        contrib_res = model.model_full_contrib(wngrid=bindown_wngrid,cutoff_grid=False)


    if args.output_file and get_rank()==0:
        from taurex.util.output import store_taurex_results,store_profiles,generate_profile_dict,generate_spectra_dict
        #Output taurex data
        with HDF5Output(args.output_file,append=True) as o:

            out = o.create_group('Output')
            if observed is not None:
                obs = o.create_group('Observed')
                observed.write(obs)

            profiles=generate_profile_dict(model)
            if observed is not None:
                spectrum= generate_spectra_dict(result,contrib_res,model.nativeWavenumberGrid,observed.wavenumberGrid)
            else:
                spectrum= generate_spectra_dict(result,contrib_res,model.nativeWavenumberGrid)

            if solution is not None:
                out.store_dictionary(solution,group_name='Solutions')
                priors = {}
                priors['Profiles'] = profiles
                priors['Spectra'] = spectrum
                out.store_dictionary(priors,group_name='Priors')
            else:
                out.store_dictionary(profiles,group_name='Profiles')
                out.store_dictionary(spectrum,group_name='Spectra')



            if optimizer:
                optimizer.write(o)






    
    wlgrid = 10000/bindown_wngrid
    if args.plot:

        if get_rank()==0  and nprocs()<=1:
            import matplotlib.pyplot as plt
            fig= plt.figure()
            ax = fig.add_subplot(1,1,1)
            #ax.set_xscale('log')
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
            if args.full_contrib:
                for k,v in contrib_res.items():
                    first_name = k
                    for out in v:
                        second_name = out[0]
                        label='{}-{}'.format(first_name,second_name)
                        if is_lightcurve:
                            binned = out[-1][1]
                            ax.plot(binned,label=label,alpha=0.6)
                        else:
                            binned = out[1]
                                
                            ax.plot(wlgrid,binned,label=label,alpha=0.6)

            
            
            #If we have an observation then plot it




            plt.legend()
            plt.show()
        else:
            logging.getLogger('taurex').warning('Number of processes > 1 so not plotting')
    



if __name__=="__main__":
    main()