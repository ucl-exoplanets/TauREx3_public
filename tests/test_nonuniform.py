import unittest
from taurex.util.fitting import validate_priors, MalformedPriorInput, parse_priors

class TestParsing(unittest.TestCase):

    def test_validation(self):

        validate_priors("Uniform(hello=1,bye=5)")

        with self.assertRaises(MalformedPriorInput):
            validate_priors("Uniform((hello=1,bye=5")
            validate_priors("Uniform(hello=1,bye=5))")
            validate_priors("Uniformhello=1,bye=5")
            validate_priors("Uniform(hello=1,bye=5")

    def test_parse(self):

        test_uniform = "Uniform(min=100,max=1532)"
        test_loguniform = "LogUniform(min=-4,max=10)"
        test_gaussian = "Gaussian(mean=200,std=30,_type='cool', bounds=(100, 789))"
        # test_loggaussian = "LogGaussian(mean=2,std=1)"

        res = parse_priors(test_uniform)
        self.assertEqual(res[0], "Uniform")
        func_args = res[1]
        self.assertIn('min', func_args)
        self.assertIn('max', func_args)
        self.assertEqual(func_args['min'], 100)
        self.assertEqual(func_args['max'], 1532)

        res = parse_priors(test_loguniform)
        self.assertEqual(res[0], "LogUniform")
        func_args = res[1]
        self.assertIn('min', func_args)
        self.assertIn('max', func_args)
        self.assertEqual(func_args['min'], -4)
        self.assertEqual(func_args['max'], 10)

        res = parse_priors(test_gaussian)
        self.assertEqual(res[0], "Gaussian")
        func_args = res[1]
        self.assertIn('mean', func_args)
        self.assertIn('std', func_args)
        self.assertIn('bounds', func_args)
        self.assertIn('_type', func_args)
        self.assertEqual(func_args['mean'], 200)
        self.assertEqual(func_args['std'], 30)
        self.assertEqual(func_args['_type'], 'cool')
        self.assertIsInstance(func_args['bounds'], tuple)
        self.assertIn(100, func_args['bounds'])
        self.assertIn(789, func_args['bounds'])