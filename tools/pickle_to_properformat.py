import pickle
import h5py




if __name__ == "__main__":
    import argparse
    import pathlib
    parser = argparse.ArgumentParser(description='pickle-to-hdf5-converter')
    parser.add_argument("-i","--pickle-file",dest="pickle",type=str,required=True,help="Pickle file to convert")
    parser.add_argument("-m","--molecule-name",dest="mol",type=str,default=None,help="(Optional) Explicit name of molecule")
    parser.add_argument("-u","--pressure-unit",dest="p_unit",type=str,default="bar",help="units for pressure (see astropy.units for compatable names")
    parser.add_argument("-o","--output",dest="output",type=str,required=True,help="Output filename")
    args=parser.parse_args()

    mol_name = args.mol

    if mol_name is None:
        splits = pathlib.Path(args.pickle).stem.split('.')
        mol_name = splits[0]
    print('Running for molecule ', mol_name)

    with h5py.File(args.output,'w') as fd:
        try:
            with open(args.pickle,'rb') as pfd:
                xsec_pickle = pickle.load(pfd)
        except UnicodeDecodeError:
            with open(args.pickle,'rb') as pfd:
                xsec_pickle = pickle.load(pfd, encoding='latin1')
        fd.create_dataset('bin_edges',data=xsec_pickle['wno'],shape=xsec_pickle['wno'].shape)
        print('stored binedges', xsec_pickle['wno'])
        fd.create_dataset('t',data=xsec_pickle['t'],shape=xsec_pickle['t'].shape)
        print('stored temp', xsec_pickle['t'])
        press = fd.create_dataset('p',data=xsec_pickle['p'],shape=xsec_pickle['p'].shape)
        fd.create_dataset('xsecarr',data=xsec_pickle['xsecarr'],shape=xsec_pickle['xsecarr'].shape)
        press.attrs['units'] = args.p_unit
        print('stored pressure', xsec_pickle['p'])
        print('Pressure units are', args.p_unit)
        fd.create_dataset('mol_name',data=mol_name)
        fd.create_dataset('key_iso_II',data='pickle')

        from taurex.opacity.hdf5opacity import HDF5Opacity
        hdf5 = HDF5Opacity(args.output)
        print('Pressure in pascal is ',hdf5.pressureGrid)
        print('Molecule name in HDF5 is',hdf5.moleculeName)

        #Testing with
