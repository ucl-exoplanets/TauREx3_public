import unittest
from unittest.mock import patch
from unittest import mock
import numpy as np
from taurex.binning.binner import Binner


class TestBinner(unittest.TestCase):


    def test_exception(self):

        binner = Binner()
        
        with self.assertRaises(NotImplementedError):
            binner.bindown(None,None)
