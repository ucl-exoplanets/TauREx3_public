
from .output import Output, OutputGroup
import h5py
from taurex.mpi import get_rank, only_master_rank
import datetime


class HDF5OutputGroup(OutputGroup):

    def __init__(self, entry):
        super().__init__('HDF5Group')
        self._entry = entry

    @only_master_rank
    def write_array(self, array_name, array, metadata=None):
        ds = self._entry.create_dataset(
            str(array_name), data=array, shape=array.shape, dtype=array.dtype)
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v

    @only_master_rank
    def write_scalar(self, scalar_name, scalar, metadata=None):
        ds = self._entry.create_dataset(str(scalar_name), data=scalar)
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v

    @only_master_rank
    def write_string(self, string_name, string, metadata=None):
        ds = self._entry.create_dataset(str(string_name), data=string)
        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v

    def create_group(self, group_name):
        entry = None
        if self._entry:
            entry = self._entry.create_group(str(group_name))
        return HDF5OutputGroup(entry)

    @only_master_rank
    def write_string_array(self, string_name, string_array, metadata=None):

        asciiList = [n.encode("ascii", "ignore") for n in string_array]
        ds = self._entry.create_dataset(
            str(string_name), (len(asciiList), 1), 'S64', asciiList)

        if metadata:
            for k, v in metadata.items():
                ds.attrs[k] = v


class HDF5Output(Output):
    def __init__(self, filename, append=False):
        super().__init__('HDF5Output')
        self.filename = filename
        self._append = append
        self.fd = None

    def open(self):
        if get_rank() == 0:
            self.fd = self._openFile(self.filename, self._append)

    def _openFile(self, fname, append):

        mode = 'w'
        if self._append:
            mode = 'a'

        fd = h5py.File(fname, mode=mode)
        fd.attrs['file_name'] = fname
        fd.attrs['file_time'] = datetime.datetime.now().isoformat()
        fd.attrs['creator'] = self.__class__.__name__
        fd.attrs['HDF5_Version'] = h5py.version.hdf5_version
        fd.attrs['h5py_version'] = h5py.version.version
        fd.attrs['program_name'] = 'TauREx'
        fd.attrs['program_version'] = 'v3.0'
        return fd

    def create_group(self, group_name):
        entry = None
        if self.fd:
            entry = self.fd.create_group(str(group_name))
        return HDF5OutputGroup(entry)

    def close(self):
        if self.fd:
            self.fd.flush()
            self.fd.close()
