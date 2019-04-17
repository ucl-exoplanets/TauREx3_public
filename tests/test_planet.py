import unittest
import numpy as np
from taurex.data.planet import Planet
from taurex.constants import G,RJUP,MJUP
 
 
class PlanetTest(unittest.TestCase):
 
    def setUp(self):
        
        self.jup = Planet(MJUP,RJUP)
        self.earth = Planet(5.972e24,6371000)
    
    def test_properties(self):
        
        self.assertEqual(self.jup.mass,MJUP)
        self.assertEqual(self.jup.radius,RJUP)
        self.assertAlmostEqual(self.jup.gravity,25.916,places=2)

        self.assertAlmostEqual(self.earth.gravity,9.819,places=2)