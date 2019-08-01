import unittest
import shutil, tempfile
from os import path
from unittest.mock import patch, mock_open
from taurex.util.math import compute_rayleigh_cross,compute_refractive_index
import numpy as np
import pickle

class ForwardModelTest(unittest.TestCase):
    
    def test_init(self):
        pass