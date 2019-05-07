"""The main taurex program"""



def main():
    import argparse

    import os
    import logging
    import numpy as np
    from taurex.parameter import ParameterParser
    import matplotlib.pyplot as plt
    from taurex.util import bindown
    logging.basicConfig(level=logging.WARNING,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Taurex')
    parser.add_argument("-i", "--input",dest='input_file',type=str,required=True,help="Input par file to pass")
    parser.add_argument("-R", "--retrieval",dest='retrieval',default=False, help="When set, runs retrieval",action='store_true')
    parser.add_argument("-p", "--plot",dest='plot',default=True,type=bool,help="Whether to plot after the run")

    args=parser.parse_args()

    #Parse the input file
    pp = ParameterParser()
    pp.read(args.input_file)

    #Setup global parameters
    pp.setup_globals()
    #Generate a model from the input
    model = pp.generate_model()

    #build the model
    model.build()

    #Get the spectrum
    observed,bindown_wngrid = pp.generate_spectrum()
    
    logging.info('Using grid {}'.format(bindown_wngrid))
    logging.info('Observed grid is {}'.format(observed.wavenumberGrid))
    #Get the native grid
    native_grid = model.nativeWavenumberGrid

    #If we bin down then get the appropriate grid
    if bindown_wngrid is None:
        bindown_wngrid = native_grid
    

    if args.retrieval is False:
        #Run the model
        absp,tau,contrib=model.model(native_grid,return_contrib=True)
        #Get out new binned down model
        new_absp = bindown(native_grid,absp,bindown_wngrid)
    else:
        if observed is None:
            logging.critical('No spectrum is defined!!')
            quit()
        
        optimizer = pp.generate_optimizer()
        optimizer.set_model(model)
        optimizer.set_observed(observed)
        optimizer.set_wavenumber_grid(bindown_wngrid)

        fitting_parameters = pp.generate_fitting_parameters()

        for key,value in fitting_parameters.items():
            fit = value['fit']
            bounds = value['bounds']

            if fit:
                logging.info('Fitting: {}'.format(key))
                optimizer.enable_fit(key)
            else:
                optimizer.disable_fit(key)
            
            if bounds:
                optimizer.set_boundary(key,bounds)

        optimizer.fit()

    #If we have an observation then plot it
    if observed is not None:
        plt.plot(np.log10(observed.wavelengthGrid),observed.spectrum,label='observed')

    wlgrid = np.log10(10000/bindown_wngrid)
    #Plot the absorption
    plt.plot(wlgrid[:-1],new_absp,label='forward model')
    for name,value in contrib:
        new_value = bindown(native_grid,value,bindown_wngrid)
        plt.plot(wlgrid[:-1],new_value,label=name)

    plt.legend()
    plt.show()
    



if __name__=="__main__":
    main()