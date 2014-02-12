import numpy as np


class FieldsTDEM(object):
    """docstring for FieldsTDEM"""

    phi0 = None #: Initial electric potential
    A0 = None #: Initial magnetic vector potential
    e0 = None #: Initial electric field
    b0 = None #: Initial magnetic flux density
    j0 = None #: Initial current density
    h0 = None #: Initial magnetic field

    phi = None #: Electric potential
    A = None #: Magnetic vector potential
    e = None #: Electric field
    b = None #: Magnetic flux density
    j = None #: Current density
    h = None #: Magnetic field

    def __init__(self, mesh, nTx, nTimes, store):

        self.nTimes = nTimes #: Number of times
        self.nTx = nTx #: Number of transmitters
        self.mesh = mesh

    def update(self, newFields, tInd):
        self.set_b(newFields['b'], tInd)
        self.set_e(newFields['e'], tInd)

    def fieldVec(self):
        u = np.ndarray((0,self.nTx))
        for i in range(self.nTimes):
            u = np.r_[u, self.get_b(i), self.get_e(i)]
        return u

    ####################################################
    # Get Methods
    ####################################################

    def get_b(self, ind):
        if ind == -1:
            return self.b0
        else:
            return self.b[ind,:,:]

    def get_e(self, ind):
        if ind == -1:
            return self.e0
        else:
            return self.e[ind,:,:]

    ####################################################
    # Set Methods
    ####################################################

    def set_b(self, b, ind):
        if self.b is None:
            self.b = np.zeros((self.nTimes, np.sum(self.mesh.nF), self.nTx))
            self.b[:] = np.nan
        self.b[ind,:,:] = b

    def set_e(self, e, ind):
        if self.e is None:
            self.e = np.zeros((self.nTimes, np.sum(self.mesh.nE), self.nTx))
            self.e[:] = np.nan
        self.e[ind,:,:] = e