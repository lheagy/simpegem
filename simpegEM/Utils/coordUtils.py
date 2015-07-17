import numpy as np

def crossProd(v0,v1):
    """
    	Cross product of 2 vectors
    """ 
    # ensure both n0, n1 are vectors of length 1
    assert len(v0) == 3, "Length of v0 should be 3"
    assert len(v1) == 3, "Length of v1 should be 3"

    v2 = np.zeros(3,dtype=float)

    v2[0] = v0[1]*v1[2] - v1[1]*v0[2]
    v2[1] = v1[0]*v0[2] - v0[0]*v1[2]
    v2[2] = v0[0]*v1[1] - v1[0]*v0[1]

    return v2

def rotationMatFromNormals(n0,n1):
    """
    	Performs the minimum number of rotations to define a rotation from the direction indicated by the vector n0 to the direction indicated by n1.
    	The axis of rotation is n0 x n1
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    """

    # ensure both n0, n1 are vectors of length 1
    assert len(n0) == 3, "Length of n0 should be 3"
    assert len(n1) == 3, "Length of n1 should be 3"

    # ensure both are true normals
    n0 *= 1./np.linalg.norm(n0)
    n1 *= 1./np.linalg.norm(n1)

    n0dotn1 = n0.dot(n1) 

    # define the rotation axis, which is the cross product of the two vectors
    rotAx = crossProd(n0,n1)
    rotAx *= 1./np.linalg.norm(rotAx)

    cosT = n0dotn1/(np.linalg.norm(n0)*np.linalg.norm(n1))
    sinT = np.sqrt(1.-n0dotn1**2)

    ux = np.array([[0., -rotAx[2], rotAx[1]], [rotAx[2], 0., -rotAx[0]], [-rotAx[1], rotAx[0], 0.]],dtype=float)

    return np.eye(3,dtype=float) + sinT*ux + (1.-cosT)*(ux.dot(ux))



