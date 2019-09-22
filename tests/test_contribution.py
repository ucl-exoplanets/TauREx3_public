import unittest
from taurex.contributions import LeeMieContribution
import numpy as np


class LeeMieTest(unittest.TestCase):

    def test_init(self):
        from taurex.model import TransmissionModel
        model = TransmissionModel()
        model.build()
        mie = LeeMieContribution()

        wngrid = np.linspace(600, 30000, 100)

        for name, xsec in mie.prepare_each(model, wngrid):
            self.assertEqual(name, 'Lee')
            self.assertEqual(wngrid.shape[0], xsec.shape[0])
