import unittest
import numpy as np
from taurex.data.fittable import fitparam, Fittable


class FittableTest(unittest.TestCase):

    def setUp(self):

        class TestClass(Fittable):
            def __init__(self):
                super().__init__()
                self._temperature = 10
                self._pressure = 50

            @fitparam(param_name='temperature', param_latex='LOL')
            def temperature(self):
                return self._temperature

            @temperature.setter
            def temperature(self, value):
                self._temperature = value

            @fitparam(param_name='pressure', param_latex='LOL')
            def pressure(self):
                return self._pressure

            @pressure.setter
            def pressure(self, value):
                self._pressure = value

        class TestClassChild(TestClass):
            def __init__(self):
                super().__init__()

        self._test_class = TestClass()
        self._test_class_child = TestClassChild()

    def test_compileparams(self):

        params = self._test_class.fitting_parameters()
        self.assertIn('temperature', params)
        self.assertIn('pressure', params)

        params = self._test_class_child.fitting_parameters()
        self.assertIn('temperature', params)
        self.assertIn('pressure', params)

    def test_read_write_params(self):
        print('Testing readwrite')
        params = self._test_class.fitting_parameters()
        temperature = params['temperature']
        pressure = params['pressure']
        self.assertEqual(temperature[2](), 10)
        self.assertEqual(pressure[2](), 50)

        temperature[3](40)
        pressure[3](80)

        self.assertEqual(self._test_class.temperature, 40)
        self.assertEqual(self._test_class.pressure, 80)

        print('Testing readwrite')
        params = self._test_class_child.fitting_parameters()
        temperature = params['temperature']
        pressure = params['pressure']
        self.assertEqual(temperature[2](), 10)
        self.assertEqual(pressure[2](), 50)

        temperature[3](40)
        pressure[3](80)

        self.assertEqual(self._test_class_child.temperature, 40)
        self.assertEqual(self._test_class_child.pressure, 80)
