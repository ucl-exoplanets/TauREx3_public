"""The main taurex program"""



def main():
    import argparse
    from taurex.parameter import ParameterParser
    import os
    import logging
    import numpy as np

    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Taurex')
    parser.add_argument("-i", "--input",dest='input_file',type=str,required=True,help="Input par file to pass")
    parser.add_argument("-R", "--retrieval",dest='retrieval',type=bool,default=False, help="True=run retrieval, False=runs forward model only")
    parser.add_argument("-p", "--plot",dest='plot',type=bool,help="Whether to plot after the run")


    args=parser.parse_args()


    pp = ParameterParser()
    pp.read(args.input_file)

    model = pp.generate_model()

    print(model)
    model.build()
    print(model.model(np.linspace(0,20000,10000)))


if __name__=="__main__":
    main()