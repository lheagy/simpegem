import unittest, os
import numpy as np
from SimPEG import Utils
from simpegEM.Utils import coordUtils

tol = 1e-15

class coorUtilsTest(unittest.TestCase):

    def test_CrossProd(self):
        self.assertTrue(all(coordUtils.crossProd(np.r_[1.,0.,0.],np.r_[0.,1.,0.]) - np.r_[0.,0.,1.] < tol))
        self.assertTrue(all(coordUtils.crossProd(np.r_[np.sqrt(2.),np.sqrt(2.),0.],np.r_[np.sqrt(2.),np.sqrt(2.),0.]) - np.r_[0.,0.,1.] < tol))
        self.assertTrue(all(coordUtils.crossProd(np.r_[0.,1.,0.],np.r_[0.,0.,1.]) - np.r_[1.,0.,0.] < tol))
        self.assertTrue(all(coordUtils.crossProd(np.r_[0.,np.sqrt(2.),np.sqrt(2.)],np.r_[0.,np.sqrt(2.),np.sqrt(2.)]) - np.r_[1.,0.,0.] < tol))
        self.assertTrue(all(coordUtils.crossProd(np.r_[0.,0.,1.],np.r_[1.,0.,0.]) - np.r_[0.,1.,0.] < tol))
        self.assertTrue(all(coordUtils.crossProd(np.r_[np.sqrt(2.),0.,np.sqrt(2.)],np.r_[np.sqrt(2.),0.,np.sqrt(2.)]) - np.r_[0.,1.,0.] < tol))

    def test_RotMat(self):
        v0 = np.random.rand(3)
        v0 *= 1./np.linalg.norm(v0)
        v1 = np.random.rand(3)
        v1 *= 1./np.linalg.norm(v1)
        Rf = coordUtils.rotationMatFromNormals(v0,v1)
        Ri = coordUtils.rotationMatFromNormals(v1,v0)

        self.assertTrue(all(Utils.mkvc(Rf.dot(v0) - v1) < tol))
        self.assertTrue(all(Utils.mkvc(Ri.dot(v1) - v0) < tol))
        

if __name__ == '__main__':
    unittest.main()

