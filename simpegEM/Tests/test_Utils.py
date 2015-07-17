import unittest, os
import numpy as np
from simpegEM.Utils import coordUtils

tol = 1e-20

class coorUtilsTest(unittest.TestCase):

    def test_CrossProd(self):
        self.assertTrue(all(coordUtils.crossProd(np.r_[1.,0.,0.],np.r_[0.,1.,0.]) - np.r_[0.,0.,1.]) < tol)
        self.assertTrue(all(coordUtils.crossProd(np.r_[np.sqrt(2.),np.sqrt(2.),0.],np.r_[np.sqrt(2.),np.sqrt(2.),0.]) - np.r_[0.,0.,1.]) < tol)
        self.assertTrue(all(coordUtils.crossProd(np.r_[0.,1.,0.],np.r_[0.,0.,1.]) - np.r_[1.,0.,0.]) < tol)
        self.assertTrue(all(coordUtils.crossProd(np.r_[0.,np.sqrt(2.),np.sqrt(2.)],np.r_[0.,np.sqrt(2.),np.sqrt(2.)]) - np.r_[1.,0.,0.]) < tol)
        self.assertTrue(all(coordUtils.crossProd(np.r_[0.,0.,1.],np.r_[1.,0.,0.]) - np.r_[0.,1.,0.]) < tol)
        self.assertTrue(all(coordUtils.crossProd(np.r_[np.sqrt(2.),0.,np.sqrt(2.)],np.r_[np.sqrt(2.),0.,np.sqrt(2.)]) - np.r_[0.,1.,0.]) < tol)

if __name__ == '__main__':
    unittest.main()

