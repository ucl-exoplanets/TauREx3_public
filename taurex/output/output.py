from taurex.log import Logger
import numpy as np


class Output(Logger):

    def __init__(self, name):
        super().__init__(name)

    def open(self):
        raise NotImplementedError

    def create_group(self, group_name):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def store_dictionary(self, dictionary, group_name=None):
        from taurex.util.util import recursively_save_dict_contents_to_output

        out = self
        if group_name is not None:
            out = self.create_group(group_name)

        recursively_save_dict_contents_to_output(out, dictionary)


class OutputGroup(Output):

    def __init__(self, name):
        super().__init__(name)
        self._name = name

    def write_array(self, array_name, array, metadata=None):
        raise NotImplementedError

    def write_list(self, list_name, list_array, metadata=None):
        arr = np.array(list_array)
        self.write_array(list_name, arr)

    def write_scalar(self, scalar_name, scalar, metadata=None):
        raise NotImplementedError

    def write_string(self, string_name, string, metadata=None):
        raise NotImplementedError

    def write_string_array(self, string_name, string_array, metadata=None):
        raise NotImplementedError
