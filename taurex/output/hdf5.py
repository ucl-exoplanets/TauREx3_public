
from .output import Output,OutputGroup
import h5py 

import datetime
class HDF5OutputGroup(OutputGroup):

    def __init__(self,entry):
        super().__init__('HDF5Group')
        self._entry = entry

    def write_array(self,array_name,array):
        self._entry.create_dataset(array_name, data=array)
    

    def write_scalar(self,scalar_name,scalar):
        self._entry.create_dataset(scalar_name, data=scalar)
    
    def write_string(self,string_name,string):
        self._entry.create_dataset(string_name, data=string)

    def create_group(self,group_name):
        entry = self._entry.create_group(group_name)
        return HDF5OutputGroup(entry)

    def write_string_array(self,string_name,string_array):

        asciiList = [n.encode("ascii", "ignore") for n in string_array]
        self._entry.create_dataset(string_name, (len(asciiList),1),'S10', asciiList)
class HDF5Output(Output):
    def __init__(self,filename):
        super().__init__('HDF5Output')
    
        self.fd = self._openFile(filename)

    def _openFile(self, fname):
        fd = h5py.File(fname, mode='w')
        fd.attrs['file_name'] = fname
        fd.attrs['file_time'] = datetime.datetime.now().isoformat()
        fd.attrs['creator'] = self.__class__.__name__
        fd.attrs['HDF5_Version'] = h5py.version.hdf5_version
        fd.attrs['h5py_version'] = h5py.version.version
        fd.attrs['program_name'] = 'TauREx'
        fd.attrs['program_version'] = 'v3.0'
        return fd
        

    def create_group(self,group_name):
        entry = self.fd.create_group(group_name)
        return HDF5OutputGroup(entry)




