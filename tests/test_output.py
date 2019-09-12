import unittest
from unittest.mock import patch
import numpy as np


class OutputDictTest(unittest.TestCase):

    contrib_list = ['Absorption','CIA','Rayleigh']

    mol_list = ['H2O','CO2','H2O2']

    def gen_contrib(self,add_extra=False):



        test_dict ={}

        for c in self.contrib_list:
            c_list =[]
            for mol in self.mol_list:

                if add_extra:
                    c_list.append((mol,np.linspace(1,10,50),np.linspace(1,10,50),np.linspace(1,10,50),('LOL','HAHa')))
                else:
                    c_list.append((mol,np.linspace(1,10,50),np.linspace(1,10,50),np.linspace(1,10,50)))
            test_dict[c] = c_list
        
        return test_dict



    def test_spectra_dict(self):
        from taurex.util.output import generate_spectra_dict
        pass
        # #Test a normal spectra

        # normal_result = (np.linspace(1,10,50),np.linspace(1,10,50),np.linspace(1,10,50),
        #         {'Absorption':{'binned':np.linspace(1,10,50),'native':np.linspace(1,10,50),'tau':np.linspace(1,10,50)},
        #         'CIA':{'binned':np.linspace(1,10,50),'native':np.linspace(1,10,50),'tau':np.linspace(1,10,50)},
        #         'Rayleigh':
        #             {'binned':np.linspace(1,10,50),'native':np.linspace(1,10,50),'tau':np.linspace(1,10,50)}

        #         })


        # spec_grid = generate_spectra_dict(normal_result,self.gen_contrib(),np.linspace(1,10,50))


        # self.assertIn('native_spectrum',spec_grid)
        # self.assertIn('native_tau',spec_grid) 
        # self.assertIn('native_wngrid',spec_grid)
        # self.assertIn('native_wlgrid',spec_grid)
        # self.assertIn('Contributions',spec_grid)

        # for c in self.contrib_list:
        #     self.assertIn(c,spec_grid['Contributions']) 
        #     for m in self.mol_list:
        #         self.assertIn(m,spec_grid['Contributions'][c])




        # self.assertNotIn('bin_wngrid',spec_grid)
        # self.assertNotIn('bin_wlgrid',spec_grid)
        # self.assertNotIn('bin_spectrum',spec_grid)
        # self.assertNotIn('bin_tau',spec_grid)


        # spec_grid = generate_spectra_dict(normal_result,self.gen_contrib(),np.linspace(1,10,50),bin_grid=np.linspace(3,8,10))

        # self.assertIn('native_spectrum',spec_grid)
        # self.assertIn('native_tau',spec_grid) 
        # self.assertIn('native_wngrid',spec_grid)
        # self.assertIn('native_wlgrid',spec_grid)
        # self.assertIn('Contributions',spec_grid)

        # for c in self.contrib_list:
        #     self.assertIn(c,spec_grid['Contributions']) 
        #     for m in self.mol_list:
        #         self.assertIn(m,spec_grid['Contributions'][c])




        # self.assertIn('bin_wngrid',spec_grid)
        # self.assertIn('bin_wlgrid',spec_grid)
        # self.assertIn('bin_spectrum',spec_grid)
        # self.assertIn('bin_tau',spec_grid)


        # contrib  = self.gen_contrib(add_extra=True)

        # spec_grid = generate_spectra_dict(normal_result,contrib,np.linspace(1,10,50),bin_grid=np.linspace(3,8,10))

        # self.assertIn('LOL',spec_grid['Contributions']['Absorption']['H2O'])