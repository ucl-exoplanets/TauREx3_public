import unittest
from taurex.parameter.classfactory import ClassFactory
from taurex.temperature import TemperatureProfile


class TestTemp(TemperatureProfile):
    pass


class ClassFactoryTest(unittest.TestCase):

    def gen_fake_module(self):
        from types import ModuleType

        m = ModuleType('test')

        m.TestTemp = TestTemp

        return m

    def test_detect_plugin_classes(self):
        cf = ClassFactory()

        fake_module = self.gen_fake_module()

        cf.load_plugin(fake_module)

        self.assertIn(TestTemp, cf.temperatureKlasses)

    def test_model_detection(self):
        from taurex.model import TransmissionModel, EmissionModel, \
            DirectImageModel
        cf = ClassFactory()

        self.assertIn(TransmissionModel, cf.modelKlasses)
        self.assertIn(EmissionModel, cf.modelKlasses)
        self.assertIn(DirectImageModel, cf.modelKlasses)
    
