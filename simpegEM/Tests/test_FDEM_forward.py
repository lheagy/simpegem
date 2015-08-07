from __future__ import division
import unittest
from SimPEG import *
from SimPEG.Tests import OrderTest
import simpegEM as EM
import sys  
from scipy.constants import mu_0
 
# for cross check test
testCrossCheck = True
testEB = True
testHJ = True

# for Fictitious Source test
testFictitiousSource = True
testType = 'Flux' # or 'Flux' or 'Field'

verbose = False

TOL = 1e-4
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = 1e-1
addrandoms = True

def getMesh(meshType='TensorMesh',n=6):
    cs = 5.
    ncx, ncy, ncz = n, n, n
    npad = 3
    if meshType is 'TensorMesh':
        hx = [(cs,npad,-1.3), (cs,ncx), (cs,npad,1.3)]
        hy = [(cs,npad,-1.3), (cs,ncy), (cs,npad,1.3)]
        hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
        mesh = Mesh.TensorMesh([hx,hy,hz],['C','C','C'])
    return mesh

def getProblem(fdemType, comp, mesh=None):

    if mesh is None: 
        mesh = getMesh()

    mapping = [('sigma', Maps.ExpMap(mesh)),('mu', Maps.IdentityMap(mesh))]

    x = np.array([np.linspace(-30,-15,3),np.linspace(15,30,3)]) #don't sample right by the source
    XYZ = Utils.ndgrid(x,x,np.r_[0.])
    Rx0 = EM.FDEM.Rx(XYZ, comp)

    SrcList = []
    SrcList.append(EM.FDEM.Src.MagDipole([Rx0], freq=freq, loc=np.r_[0.,0.,0.]))
    SrcList.append(EM.FDEM.Src.MagDipole_Bfield([Rx0], freq=freq, loc=np.r_[0.,0.,0.]))
    SrcList.append(EM.FDEM.Src.CircularLoop([Rx0], freq=freq, loc=np.r_[0.,0.,0.]))

    if verbose:
        print '  Fetching %s problem' % (fdemType)

    if fdemType == 'e':
        S_m = np.zeros(mesh.nF)
        S_e = np.zeros(mesh.nE)
        S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1.
        S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1.
        SrcList.append(EM.FDEM.Src.RawVec([Rx0], freq, S_m, S_e))

        survey = EM.FDEM.SurveyFDEM(SrcList)
        prb = EM.FDEM.ProblemFDEM_e(mesh, mapping=mapping)

    elif fdemType == 'b':
        S_m = np.zeros(mesh.nF)
        S_e = np.zeros(mesh.nE)
        S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1.
        S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1.
        SrcList.append(EM.FDEM.Src.RawVec([Rx0], freq, S_m, S_e))

        survey = EM.FDEM.SurveyFDEM(SrcList)
        prb = EM.FDEM.ProblemFDEM_b(mesh, mapping=mapping)

    elif fdemType == 'j': 
        S_m = np.zeros(mesh.nE)
        S_e = np.zeros(mesh.nF)
        S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1.
        S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1.
        SrcList.append(EM.FDEM.Src.RawVec([Rx0], freq, S_m, S_e))

        survey = EM.FDEM.SurveyFDEM(SrcList)
        prb = EM.FDEM.ProblemFDEM_j(mesh, mapping=mapping)

    elif fdemType == 'h':
        S_m = np.zeros(mesh.nE)
        S_e = np.zeros(mesh.nF)
        S_m[Utils.closestPoints(mesh,[0.,0.,0.],'Ez') + np.sum(mesh.vnE[:1])] = 1.
        S_e[Utils.closestPoints(mesh,[0.,0.,0.],'Fz') + np.sum(mesh.vnF[:1])] = 1.
        SrcList.append(EM.FDEM.Src.RawVec([Rx0], freq, S_m, S_e))

        survey = EM.FDEM.SurveyFDEM(SrcList)
        prb = EM.FDEM.ProblemFDEM_h(mesh, mapping=mapping)

    else:
        raise NotImplementedError()
    prb.pair(survey)

    try:
        from pymatsolver import MumpsSolver
        prb.Solver = MumpsSolver
    except ImportError, e:
        pass

    return prb


def crossCheckTest(fdemType, comp):

    l2norm = lambda r: np.sqrt(r.dot(r))

    prb1 = getProblem(fdemType, comp)
    mesh = prb1.mesh
    print 'Cross Checking Forward: %s formulation - %s' % (fdemType, comp)
    m = np.log(np.ones(mesh.nC)*CONDUCTIVITY)
    mu = np.log(np.ones(mesh.nC)*MU)

    if addrandoms is True:
        m  = m + np.random.randn(mesh.nC)*np.log(CONDUCTIVITY)*1e-1 
        mu = mu + np.random.randn(mesh.nC)*MU*1e-1

    survey1 = prb1.survey
    d1 = survey1.dpred(np.r_[m,mu])

    if verbose:
        print '  Problem 1 solved'

    if fdemType == 'e':   prb2 = getProblem('b', comp)
    elif fdemType == 'b': prb2 = getProblem('e', comp)
    elif fdemType == 'j': prb2 = getProblem('h', comp)
    elif fdemType == 'h': prb2 = getProblem('j', comp)
    else:
        raise NotImplementedError()
    
    survey2 = prb2.survey
    d2 = survey2.dpred(np.r_[m,mu])

    if verbose:
        print '  Problem 2 solved'

    r = d2-d1
    l2r = l2norm(r) 

    tol = np.max([TOL*(10**int(np.log10(l2norm(d1)))),FLR]) 
    print l2norm(d1), l2norm(d2),  l2r , tol, l2r < tol
    return l2r < tol    


class FDEM_ForwardTests(unittest.TestCase):

    if testCrossCheck:
        if testEB:
            def test_EB_CrossCheck_exr_Eform(self):
                self.assertTrue(crossCheckTest('e', 'exr'))
            def test_EB_CrossCheck_eyr_Eform(self):
                self.assertTrue(crossCheckTest('e', 'eyr'))
            def test_EB_CrossCheck_ezr_Eform(self):
                self.assertTrue(crossCheckTest('e', 'ezr'))
            def test_EB_CrossCheck_exi_Eform(self):
                self.assertTrue(crossCheckTest('e', 'exi'))
            def test_EB_CrossCheck_eyi_Eform(self):
                self.assertTrue(crossCheckTest('e', 'eyi'))
            def test_EB_CrossCheck_ezi_Eform(self):
                self.assertTrue(crossCheckTest('e', 'ezi'))

            def test_EB_CrossCheck_bxr_Eform(self):
                self.assertTrue(crossCheckTest('e', 'bxr'))
            def test_EB_CrossCheck_byr_Eform(self):
                self.assertTrue(crossCheckTest('e', 'byr'))
            def test_EB_CrossCheck_bzr_Eform(self):
                self.assertTrue(crossCheckTest('e', 'bzr'))
            def test_EB_CrossCheck_bxi_Eform(self):
                self.assertTrue(crossCheckTest('e', 'bxi'))
            def test_EB_CrossCheck_byi_Eform(self):
                self.assertTrue(crossCheckTest('e', 'byi'))
            def test_EB_CrossCheck_bzi_Eform(self):
                self.assertTrue(crossCheckTest('e', 'bzi'))

        if testHJ:
            def test_HJ_CrossCheck_jxr_Jform(self):
                self.assertTrue(crossCheckTest('j', 'jxr'))
            def test_HJ_CrossCheck_jyr_Jform(self):
                self.assertTrue(crossCheckTest('j', 'jyr'))
            def test_HJ_CrossCheck_jzr_Jform(self):
                self.assertTrue(crossCheckTest('j', 'jzr'))
            def test_HJ_CrossCheck_jxi_Jform(self):
                self.assertTrue(crossCheckTest('j', 'jxi'))
            def test_HJ_CrossCheck_jyi_Jform(self):
                self.assertTrue(crossCheckTest('j', 'jyi'))
            def test_HJ_CrossCheck_jzi_Jform(self):
                self.assertTrue(crossCheckTest('j', 'jzi'))

            def test_HJ_CrossCheck_hxr_Jform(self):
                self.assertTrue(crossCheckTest('j', 'hxr'))
            def test_HJ_CrossCheck_hyr_Jform(self):
                self.assertTrue(crossCheckTest('j', 'hyr'))
            def test_HJ_CrossCheck_hzr_Jform(self):
                self.assertTrue(crossCheckTest('j', 'hzr')) 
            def test_HJ_CrossCheck_hxi_Jform(self):
                self.assertTrue(crossCheckTest('j', 'hxi'))
            def test_HJ_CrossCheck_hyi_Jform(self):
                self.assertTrue(crossCheckTest('j', 'hyi'))
            def test_HJ_CrossCheck_hzi_Jform(self):
                self.assertTrue(crossCheckTest('j', 'hzi'))


class fictitiousSourceTest(OrderTest): 
    name = "Fictitious Source"
    meshTypes = ['uniformTensorMesh', 'uniformCurv']
    meshSizes = [2, 4, 8]
    freq = 1. 



    def getError(self):

        np.random.seed = 2
        a = 1e10*np.random.rand(6)
        np.random.seed = 5
        b = np.random.rand(6) + 10. # make sure b is large enough so that neither sig nor mu is below zero
        c = 20.

        r2 = lambda x, y, z: x**2 + y**2 + z**2
        r = lambda x, y, z: np.sqrt(r)

        muFun = lambda x, y, z: mu_0 / ( (np.arctan(a[0]*x) + b[0])*(np.arctan(a[1]*y) + b[1])*(np.arctan(a[2]*z) + b[2]))
        sigFun = lambda x, y, z: 1. / ( (np.arctan(a[3]*x) + b[3])*(np.arctan(a[4]*y) + b[4])*(np.arctan(a[5]*z) + b[5]))

        exFun = lambda x, y, z: np.exp(-r2(x, y, z)/c) * (np.arctan(a[0]*x) + b[0])
        eyFun = lambda x, y, z: np.exp(-r2(x, y, z)/c) * (np.arctan(a[1]*y) + b[1])
        ezFun = lambda x, y, z: np.exp(-r2(x, y, z)/c) * (np.arctan(a[2]*z) + b[2])

        hxFun = lambda x, y, z: np.exp(-r2(x, y, z)/c) * (np.arctan(a[3]*x) + b[3])
        hyFun = lambda x, y, z: np.exp(-r2(x, y, z)/c) * (np.arctan(a[4]*y) + b[4])
        hzFun = lambda x, y, z: np.exp(-r2(x, y, z)/c) * (np.arctan(a[5]*z) + b[5])

        jxFun = lambda x, y, z: sigFun(x, y, z) * exFun(x, y, z)
        jyFun = lambda x, y, z: sigFun(x, y, z) * eyFun(x, y, z)
        jzFun = lambda x, y, z: sigFun(x, y, z) * ezFun(x, y, z)

        bxFun = lambda x, y, z: muFun(x, y, z) * hxFun(x, y, z)
        byFun = lambda x, y, z: muFun(x, y, z) * hyFun(x, y, z)
        bzFun = lambda x, y, z: muFun(x, y, z) * hzFun(x, y, z)


        def E(gridX, gridY, gridZ):

            ex = exFun(gridX[:,0], gridX[:,1], gridX[:,2])
            ey = eyFun(gridY[:,0], gridY[:,1], gridY[:,2])
            ez = ezFun(gridZ[:,0], gridZ[:,1], gridZ[:,2])

            return Utils.mkvc(np.hstack([ex, ey, ez]),2)

        def B(gridX, gridY, gridZ):

            bx = bxFun(gridX[:,0], gridX[:,1], gridX[:,2])
            by = byFun(gridY[:,0], gridY[:,1], gridY[:,2])
            bz = bzFun(gridZ[:,0], gridZ[:,1], gridZ[:,2])

            return Utils.mkvc(np.hstack([bx, by, bz]),2)

        def J(gridX, gridY, gridZ):

            jx = jxFun(gridX[:,0], gridX[:,1], gridX[:,2])
            jy = jyFun(gridY[:,0], gridY[:,1], gridY[:,2])
            jz = jzFun(gridZ[:,0], gridZ[:,1], gridZ[:,2])

            return Utils.mkvc(np.hstack([jx, jy, jz]),2)

        def H(gridX, gridY, gridZ):

            hx = hxFun(gridX[:,0], gridX[:,1], gridX[:,2])
            hy = hyFun(gridY[:,0], gridY[:,1], gridY[:,2])
            hz = hzFun(gridZ[:,0], gridZ[:,1], gridZ[:,2])

            return Utils.mkvc(np.hstack([hx, hy, hz]),2)

        def curlE(gridX, gridY, gridZ):

            dy_exFun = lambda x, y, z: -2./c * y * exFun(x, y, z)
            dz_exFun = lambda x, y, z: -2./c * z * exFun(x, y, z)
            dx_eyFun = lambda x, y, z: -2./c * x * eyFun(x, y, z)
            dz_eyFun = lambda x, y, z: -2./c * z * eyFun(x, y, z)
            dx_ezFun = lambda x, y, z: -2./c * x * ezFun(x, y, z)
            dy_ezFun = lambda x, y, z: -2./c * y * ezFun(x, y, z) 

            curl_eFun_x = lambda x, y, z: dy_ezFun(x, y, z) - dz_eyFun(x, y, z)
            curl_eFun_y = lambda x, y, z: dz_exFun(x, y, z) - dx_ezFun(x, y, z)
            curl_eFun_z = lambda x, y, z: dx_eyFun(x, y, z) - dy_exFun(x, y, z)

            curlE_x = curl_eFun_x(gridX[:,0], gridX[:,1], gridX[:,2])
            curlE_y = curl_eFun_y(gridY[:,0], gridY[:,1], gridY[:,2])
            curlE_z = curl_eFun_z(gridZ[:,0], gridZ[:,1], gridZ[:,2])

            return Utils.mkvc(np.hstack([curlE_x, curlE_y, curlE_z]),2)


        def curlH(gridX, gridY, gridZ): 

            dy_hxFun = lambda x, y, z: -2./c * y * hxFun(x, y, z)
            dz_hxFun = lambda x, y, z: -2./c * z * hxFun(x, y, z)
            dx_hyFun = lambda x, y, z: -2./c * x * hyFun(x, y, z)
            dz_hyFun = lambda x, y, z: -2./c * z * hyFun(x, y, z)
            dx_hzFun = lambda x, y, z: -2./c * x * hzFun(x, y, z)
            dy_hzFun = lambda x, y, z: -2./c * y * hzFun(x, y, z) 

            curl_hFun_x = lambda x, y, z: dy_hzFun(x, y, z) - dz_hyFun(x, y, z)
            curl_hFun_y = lambda x, y, z: dz_hxFun(x, y, z) - dx_hzFun(x, y, z)
            curl_hFun_z = lambda x, y, z: dx_hyFun(x, y, z) - dy_hxFun(x, y, z)

            curlH_x = curl_hFun_x(gridX[:,0], gridX[:,1], gridX[:,2])
            curlH_y = curl_hFun_y(gridY[:,0], gridY[:,1], gridY[:,2])
            curlH_z = curl_hFun_z(gridZ[:,0], gridZ[:,1], gridZ[:,2])

            return Utils.mkvc(np.hstack([curlH_x, curlH_y, curlH_z]),2)

        fdemType = self.fdemType

        mu = muFun(self.M.gridCC[:,0],self.M.gridCC[:,1],self.M.gridCC[:,2])
        sigma = sigFun(self.M.gridCC[:,0],self.M.gridCC[:,1],self.M.gridCC[:,2])

        if fdemType is 'e' or fdemType is 'b':
            S_mfict = curlE(self.M.gridFx, self.M.gridFy, self.M.gridFz) + EM.Utils.EMUtils.omega(freq) * B(self.M.gridFx, self.M.gridFy, self.M.gridFz)
            
            S_efict = curlH(self.M.gridEx, self.M.gridEy, self.M.gridEz) - J(self.M.gridEx, self.M.gridEy, self.M.gridEz)

        elif fdemType is 'h' or fdemType is 'j':
            S_mfict = curlE(self.M.gridEx, self.M.gridEy, self.M.gridEz) + EM.Utils.EMUtils.omega(freq) * B(self.M.gridEx, self.M.gridEy, self.M.gridEz)
            
            S_efict = curlH(self.M.gridFx, self.M.gridFy, self.M.gridFz) - J(self.M.gridFx, self.M.gridFy, self.M.gridFz)


        mapping = [('sigma', Maps.IdentityMap(self.M)),('mu', Maps.IdentityMap(self.M))]
  
        src = EM.FDEM.Src.RawVec([], freq, S_mfict, S_efict)
        survey = EM.FDEM.SurveyFDEM([src])

        if fdemType is 'e': 
            prb = EM.FDEM.ProblemFDEM_e(self.M, mapping=mapping)

            if testType is 'Flux':  
                sol_ana = B(self.M.gridFx, self.M.gridFy, self.M.gridFz)
                Av = self.M.aveF2CCV
                solType = 'b'
            elif testType is 'Field':
                sol_ana = E(self.M.gridEx, self.M.gridEy, self.M.gridEz)
                solType = 'e'
                Av = self.M.aveE2CCV

        elif fdemType is 'b':
            prb = EM.FDEM.ProblemFDEM_b(self.M, mapping=mapping)

            if testType is 'Flux':  
                sol_ana = B(self.M.gridFx, self.M.gridFy, self.M.gridFz)
                Av = self.M.aveF2CCV
                solType = 'b'
            elif testType is 'Field':
                sol_ana = E(self.M.gridEx, self.M.gridEy, self.M.gridEz)
                solType = 'e'
                Av = self.M.aveE2CCV

        elif fdemType is 'j': 
            prb = EM.FDEM.ProblemFDEM_j(self.M, mapping=mapping)

            if testType is 'Flux':  
                sol_ana = J(self.M.gridFx, self.M.gridFy, self.M.gridFz)
                Av = self.M.aveF2CCV
                solType = 'j'
            elif testType is 'Field':
                sol_ana = H(self.M.gridEx, self.M.gridEy, self.M.gridEz)
                solType = 'h'
                Av = self.M.aveE2CCV

        elif fdemType is 'h': 
            prb = EM.FDEM.ProblemFDEM_h(self.M, mapping=mapping)

            if testType is 'Flux':  
                sol_ana = J(self.M.gridFx, self.M.gridFy, self.M.gridFz)
                Av = self.M.aveF2CCV
                solType = 'j'
            elif testType is 'Field':
                sol_ana = H(self.M.gridEx, self.M.gridEy, self.M.gridEz)
                solType = 'h'
                Av = self.M.aveE2CCV

        prb.pair(survey)

        try:
            from pymatsolver import MumpsSolver
            prb.Solver = MumpsSolver
        except ImportError, e:
            pass

        sol = prb.fields(np.r_[sigma,mu])[src,solType]

        res2 = abs(sol - sol_ana)**2
        Vt = np.kron(np.ones([1,3]),self.M.vol)
        err = np.sqrt(Vt*(Av*res2))[0][0]
        return err

    def test_FictSource_e(self):
        self.name = 'Ficticious Source Test e'
        self.fdemType = 'e'      
        self.orderTest()
    def test_FictSource_b(self):
        self.name = 'Ficticious Source Test b'
        self.fdemType = 'b'      
        self.orderTest()
    def test_FictSource_h(self):
        self.name = 'Ficticious Source Test h'
        self.fdemType = 'h'      
        self.orderTest()
    def test_FictSource_j(self):
        self.name = 'Ficticious Source Test j'
        self.fdemType = 'j'      
        self.orderTest()


if __name__ == '__main__':
    unittest.main()
