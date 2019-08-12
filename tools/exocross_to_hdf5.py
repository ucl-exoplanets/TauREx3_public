


import argparse
import glob
import h5py
import os
import argparse
import pandas as pd
def run_conversion(*args):
    
    temp,press,f = args[0]
    df = pd.read_csv(f,delim_whitespace=True,usecols=[1],header=None)
    return temp,press,f,df[1].values



def xsec_iterator(xsec_list,sep,t_pos,p_pos):
    import re
    import pathlib
    regex = re.compile("\d+\.*\d+")
    for xsec in xsec_list:
        clean_path = pathlib.Path(xsec).stem
        split = clean_path.split(sep)

        T = float(regex.search(split[t_pos]).group(0))
        P = float(regex.search(split[p_pos]).group(0))
        #split=xsec.split(sep)
        #T = regex.search(split[t_pos])
        #P = regex.search(split[p_pos])
        yield T,P,xsec
        
        #float(P.group(0)),float(T.group(0))




def main():
    import numpy as np
    import concurrent.futures
    parser = argparse.ArgumentParser(description='Exocross-to-hdf5-converter')
    parser.add_argument("-d", "--xsec-dir",dest='dir',type=str,required=True,help="directory containing Exocross xsec outputs")
    parser.add_argument("-s","--seperator",dest="sep",type=str,default="_",help="seperator to split filename string by")
    parser.add_argument("-t","--temperature-pos",dest="T",type=int,default=2,help="Position of temperature in string array once split")
    parser.add_argument("-p","--pressure-pos",dest="P",type=int,default=3,help="Position of pressure in string array once split")
    parser.add_argument("-m","--molecule-name",dest="mol",type=str,default=None,help="(Optional) Explicit name of molecule")

    parser.add_argument("-u","--pressure-unit",dest="p_unit",type=str,default="bar",help="units for pressure (see astropy.units for compatable names")
    parser.add_argument("-o","--output",dest="output",type=str,required=True,help="Output filename")
    parser.add_argument("-n","--num-threads",dest="nthreads",type=int,default=2,help="number of cores to use in conversion")
    args=parser.parse_args()

    xsec_list = glob.glob(os.path.join(args.dir,'*.xsec'))
    
    mol_name = args.mol
    if args.mol is None:
        import pathlib
        mol_name = pathlib.Path(xsec_list[0]).stem.split(args.sep)[0]
    
    
    print('Molecule is {}'.format(mol_name))

    #Collect temperature and pressure list
    temp_list = []
    press_list = []
    #Get the temperature list and pressure list
    for T,P,f in xsec_iterator(xsec_list,args.sep,args.T,args.P):
        temp_list.append(T)
        press_list.append(P)

    temp_list = np.sort(np.array(list(set(temp_list))))
    press_list = np.sort(np.array(list(set(press_list))))

    print('Found temperatures:')
    print(temp_list)
    print('Found pressures:')
    print(press_list)


    #Now get the bin edges
    bin_edges = np.loadtxt(xsec_list[0],usecols=(0)).reshape(-1)

    print ('Determined bin edges: {}'.format(bin_edges))

    xsecarr_shape = (len(press_list),len(temp_list),len(bin_edges))
    print ('Cross-section grid is shaped {}'.format(xsecarr_shape))
    #Now lets write it!!!
    print('creating file ',args.output)

    num_files = len(xsec_list)
    files_process= 0
    with h5py.File(args.output,'w') as fd:
        import concurrent
        fd.create_dataset('bin_edges',data=bin_edges,shape=bin_edges.shape)
        fd.create_dataset('mol_name',data=mol_name)
        fd.create_dataset('key_iso_II',data='exocross')
        press = fd.create_dataset('p',data=press_list)
        press.attrs['units'] = args.p_unit
        fd.create_dataset('t',data=temp_list)

        xsecarr = fd.create_dataset('xsecarr',dtype=np.float64,shape=xsecarr_shape)

        with concurrent.futures.ProcessPoolExecutor(args.nthreads) as executor: 
            for T,P,f,xsec in executor.map(run_conversion,  xsec_iterator(xsec_list,args.sep,args.T,args.P)):

                files_process+=1
                print('Reading in {}/{}'.format(files_process,num_files)) 
                temp_idx = temp_list.searchsorted(T)
                press_idx = press_list.searchsorted(P)
                xsecarr[press_idx,temp_idx,:] =  xsec

        # for T,P,f in xsec_iterator(xsec_list,args.sep,args.T,args.P):
        #     files_process+=1
        #     print('Reading in {}/{}'.format(files_process,num_files)) 
        #     temp_idx = temp_list.searchsorted(T)
        #     press_idx = press_list.searchsorted(P)
        #     df = pd.read_csv(f,delim_whitespace=True,usecols=[1],header=None)
        #     #return temp,press,f,df[1].values
        #     xsecarr[press_idx,temp_idx,:] = df[1].values
            



    
    







if __name__=="__main__":
    main()





