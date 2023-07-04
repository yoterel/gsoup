import numpy as np
from .core import to_44, compose_rt, normalize

def sincos(a, degrees=True):
    """
    sin and cos of an angle
    :param a: angle
    :param degrees: if True, a is in degrees
    """
    if degrees:
        a = np.deg2rad(a)
    return np.sin(a), np.cos(a)

def translate(t):
    """
    creates a translation matrix from a translation vector
    :param t: translation vector or batch of translation vectors
    :return: 4x4 translation matrix
    """
    if t.shape[-1] != 3:
        raise ValueError("translation vector must be 3d")
    if t.ndim == 1:
        t = t[None, :]
    mat = np.concatenate((np.eye(3)[None, :, :], t[:, :, None]), axis=-1)
    return to_44(mat)

def scale(s):
    """
    creates a scaling matrix from a scaling vector
    :param s: scaling vector or batch of scaling vectors
    :return: nx4x4 scaling matrix
    """
    if s.shape[-1] != 3:
        raise ValueError("translation vector must be 3d")
    if s.ndim == 1:
        s = s[None, :]
    mat = np.diag(s)
    return to_44(mat)

def rotate(a, r):
    """
    creates a rotation matrix from an angle a and an axis of rotation r
    """
    s, c = sincos(a)
    r = normalize(r)
    nc = 1 - c
    x, y, z = r
    return np.array([[x*x*nc +   c, x*y*nc - z*s, x*z*nc + y*s, 0],
                      [y*x*nc + z*s, y*y*nc +   c, y*z*nc - x*s, 0],
                      [x*z*nc - y*s, y*z*nc + x*s, z*z*nc +   c, 0],
                      [           0,            0,            0, 1]])
def rotx(a, degrees=True):
    """
    creates a homogeneous 3D rotation matrix around the x axis
    a: angle
    degrees: if True, a is in degrees, else radians
    """
    s, c = sincos(a, degrees)
    return np.array([[1,0,0,0],
                      [0,c,-s,0],
                      [0,s,c,0],
                      [0,0,0,1]])

def roty(a, degrees=True):
    """
    creates a homogeneous 3D rotation matrix around the y axis
    a: angle
    degrees: if True, a is in degrees, else radians
    """
    s, c = sincos(a, degrees)
    return np.array([[c,0,s,0],
                      [0,1,0,0],
                      [-s,0,c,0],
                      [0,0,0,1]])

def rotz(a, degrees=True):
    """
    creates a homogeneous 3D rotation matrix around the z axis
    a: angle
    degrees: if True, a is in degrees, else radians
    """
    s, c = sincos(a, degrees)
    return np.array([[c,-s,0,0],
                      [s,c,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])

def find_rigid_transform(A, B):
    """
    finds best 3d rigid transformation between pc a and pc b (in the least squares sense) based on "Least-Squares Rigid Motion Using SVD"
    :param A: point cloud a
    :param B: point cloud b
    :return: W such that W@A = B
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_mean = A - centroid_A
    B_mean = B - centroid_B

    H = A_mean.T @ B_mean
    U, S, Vt = np.linalg.svd(H)

    flip = np.linalg.det(Vt.T @ U.T)
    ones = np.identity(len(Vt))
    ones[-1, -1] = flip
    R = Vt.T @ ones @ U.T
    t = centroid_B - R @ centroid_A
    return compose_rt(R[None, :], t[None, :], square=True)

def find_affine_transformation(A, B):
    """
    finds best 3d affine transformation between pc a and pc b (in the least squares sense)
    :param A: point cloud a
    :param B: point cloud b
    :return: W such that W@A = B
    """
    # assemble A matrix for least squares
    assert len(A) >= 4
    new_A = np.c_[A, np.ones(len(A))]
    new_A = np.repeat(np.tile(new_A, (1, 3)), 3, axis=0)
    # now A is 3nx12
    # zero columns depending on index of point
    new_A[::3, 4:] = 0
    new_A[1::3, :4] = 0
    new_A[1::3, 8:] = 0
    new_A[2::3, :8] = 0
    new_B = B.flatten()
    if len(A) == 4:
        #solve linear system
        W = np.linalg.inv(new_A) @ new_B
    else:
        W = np.linalg.lstsq(new_A, new_B, rcond=None)[0]
    return to_44(W.reshape(3, 4))

def decompose_affine(A44):
    """
    Decompose 4x4 homogenous affine matrix into parts.
    The parts are translations, rotations, zooms, shears.
    Returns
    -------
    T : array, shape (3,)
       Translation vector
    R : array shape (3,3)
        rotation matrix
    Z : array, shape (3,)
       scale vector.  May have one negative zoom to prevent need for negative
       determinant R matrix above
    S : array, shape (3,)
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix (type ``striu``).
    Implementation of "Decomposing a matrix into simple transformations".
    """
    A44 = np.asarray(A44)
    T = A44[:-1,-1]
    RZS = A44[:-1,:-1]
    # compute scales and shears
    M0, M1, M2 = np.array(RZS).T
    # extract x scale and normalize
    sx = np.sqrt(np.sum(M0**2))
    M0 /= sx
    # orthogonalize M1 with respect to M0
    sx_sxy = np.dot(M0, M1)
    M1 -= sx_sxy * M0
    # extract y scale and normalize
    sy = np.sqrt(np.sum(M1**2))
    M1 /= sy
    sxy = sx_sxy / sx
    # orthogonalize M2 with respect to M0 and M1
    sx_sxz = np.dot(M0, M2)
    sy_syz = np.dot(M1, M2)
    M2 -= (sx_sxz * M0 + sy_syz * M1)
    # extract z scale and normalize
    sz = np.sqrt(np.sum(M2**2))
    M2 /= sz
    sxz = sx_sxz / sx
    syz = sy_syz / sy
    # Reconstruct rotation matrix, ensure positive determinant
    Rmat = np.array([M0, M1, M2]).T
    if np.linalg.det(Rmat) < 0:
        sx *= -1
        Rmat[:,0] *= -1
    return T, Rmat, np.array([sx, sy, sz]), np.array([sxy, sxz, syz])