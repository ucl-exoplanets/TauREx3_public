"""The main taurex program"""



def main():
    import argparse

    import os
    import logging
    import numpy as np
    from taurex.parameter import ParameterParser
    import matplotlib.pyplot as plt
    from taurex.util import bindown
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Taurex')
    parser.add_argument("-i", "--input",dest='input_file',type=str,required=True,help="Input par file to pass")
    parser.add_argument("-R", "--retrieval",dest='retrieval',type=bool,default=False, help="True=run retrieval, False=runs forward model only")
    parser.add_argument("-p", "--plot",dest='plot',type=bool,help="Whether to plot after the run")

    args=parser.parse_args()


    pp = ParameterParser()
    pp.read(args.input_file)
    pp.setup_globals()
    model = pp.generate_model()

    model.build()

    observed,bindown_wngrid = pp.generate_spectrum()

    native_grid = model.nativeWavenumberGrid

    if bindown_wngrid is None:
        bindown_wngrid = native_grid
    
    absp,tau,contrib=model.model(native_grid,return_contrib=True)


    new_absp = bindown(native_grid,absp,bindown_wngrid)
    

    if observed is not None:
        plt.plot(np.log10(observed.wavelengthGrid),observed.spectrum)

    plt.plot(np.log10(10000/bindown_wngrid[:-1]),new_absp)
    plt.show()
    

if __name__=="__main__":
    main()