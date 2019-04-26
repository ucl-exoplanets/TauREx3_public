import unittest
import numpy as np
from taurex.data.fittable import fitparam
 
 
class FitParamTest(unittest.TestCase):


    def test_fitparam_read_write(self):
        
        class TestClass(object):
            def __init__(self):
                self._value = 10

            @fitparam(param_name='MyParam',param_latex='TEST')
            def value(self):
                return self._value
            
            @value.setter
            def value(self,v):
                self._value = v
        
        print('Init')
        test = TestClass()
        print('Finished init')
        self.assertEqual(test.value,10)

        test.value = 40
        self.assertEqual(test.value,40)
    
    def test_parameters(self):
        class TestClass(object):
            def __init__(self):
                self._value = 10

            @fitparam(param_name='MyParam',param_latex='TEST')
            def value(self):
                return self._value
            
            @value.setter
            def value(self,v):
                self._value = v
        

        test = TestClass()


        func = test.__class__.__dict__['value'].fget
        self.assertEqual(func.param_name,'MyParam')
        self.assertEqual(func.param_latex,'TEST') 


            




    