from taurex.log import Logger
import numpy as np


class Output(Logger):

    def __init__(self,name):
        super().__init__(name)

    def create_group(self,group_name):
        raise NotImplementedError



    
class OutputGroup(Output):

    def __init__(self,name):
        super().__init__(name)
        self._name = name
    

    def write_array(self,array_name,array):
        raise NotImplementedError
    
    def write_list(self,list_name,list_array):
        arr = np.array(list_array)
        self.write_array(list_name,arr)

    def write_scalar(self,scalar_name,scalar):
        raise NotImplementedError
    
    def write_string(self,string_name,string):
        raise NotImplementedError
    
    def write_string_array(self,string_name,string_array):
        raise NotImplementedError