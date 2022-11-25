import torch
import numpy as np
from PIL import Image

def normalize(x, eps=1e-7):
    if type(x) == torch.Tensor:
        return x / (torch.norm(x, dim=-1, keepdim=True) + eps)
    elif type(x) == np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def broadcast_batch(*args):
    """
    broadcast a list of arrays to the same shape on the batch dimnesion
    assumes first dimension is batch unless ndim = 1 for all inputs (but then does not broadcast)
    :param args: list of arrays
    :return: list of arrays with the same shape
    """
    shapes = [a.shape for a in args]
    ndims = np.array([a.ndim for a in args])
    new_args = []
    if (ndims <= 1).all():
        for a in args:
            new_args.append(a[None, :])
        return new_args
    elif ndims.min() == 1:
        raise ValueError("cannot broadcast 1d and nd arrays")
    else:
        batch_dim = np.array([s[0] for s in shapes]).max()
        for i, a in enumerate(args):
            if a.shape[0] == batch_dim:
                new_args.append(a)
            else:
                new_args.append(np.broadcast_to(a, (batch_dim, *a.shape[1:])))
    return new_args


def compose_rt(R: np.array, t: np.array):
    """
    composes a n x 3 x 4 numpy array from rotation and translation.
    will broadcast upon batch dimension if necessary.
    :param R: nx3x3 numpy array
    :param t: nx3 numpy array
    :return: n x 3 x 4 composition of the rotation and translation
    """
    RR, tt = broadcast_batch(R, t)
    return np.concatenate((RR, tt[:, :, None]), axis=-1)

def to_44(mat: np.array):
    """
    converts a 3x4 to a 4x4 matrix by concatenating 0 0 0 1
    :param mat: 3x4 numpy array
    :return: 4x4 numpy array
    """
    if mat.ndim == 3:
        assert mat.shape[1:] == (3, 4)
        to_cat = np.broadcast_to(np.array([0, 0, 0, 1]), (mat.shape[0], 1, 4))
        new_mat = np.concatenate((mat, to_cat), axis=1)
    else:
        assert mat.shape == (3, 4)
        new_mat = np.concatenate((mat, np.array([0, 0, 0, 1])[None, :]), axis=0)
    return new_mat

def look_at(from_, to_, up_):
    """
    returns a batch of look_at transforms 4x4 (c2w)
    will broadcast upon batch dimension if necessary.
    :param from_: n x 3 from vectors in world space
    :param to_: n x 3 at vector in world space
    :param up_: n x 3 up vector in world space
    :return: n x 4 x 4 transformation matrices (c2w)
    """
    from_, to_, up_ = broadcast_batch(from_, to_, up_)
    forward = to_[None, :] - from_
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
    right = np.cross(up_[None, :], forward)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)
    rot = np.concatenate((right[:, :, None], up[:, :, None], forward[:, :, None]), axis=-1)
    c2w = np.concatenate((rot, from_[:, :, None]), axis=-1)
    return to_44(c2w)

def look_at_torch(
        eye:torch.Tensor, #3
        at:torch.Tensor, #3
        up:torch.Tensor, #3
        device:torch.device
    ) -> torch.Tensor: #4,4
    """
    creates a lookat transform matrix
    """
    z = (at - eye).type(torch.float32).to(device)
    z /= torch.norm(z)
    x = torch.cross(up, z).type(torch.float32).to(device)
    x /= torch.norm(x)
    y = torch.cross(z, x).type(torch.float32).to(device)
    y /= torch.norm(y)
    T = torch.eye(4, device=device)
    T[:3,:3] = torch.stack([x,y,z],dim=1)
    T[:3,3] = eye
    return T

def orthographic_projection(l, r, b, t, n, f):
    dx = r - l
    dy = t - b
    dz = f - n
    rx = -(r + l) / (r - l)
    ry = -(t + b) / (t - b)
    rz = -(f + n) / (f - n)
    return np.array([[2.0/dx,0,0,rx],
                      [0,2.0/dy,0,ry],
                      [0,0,-2.0/dz,rz],
                      [0,0,0,1]])

def perspective_projection(fovy=45, aspect=1.0, n=0.1, f=100.0):
    s = 1.0/np.tan(np.deg2rad(fovy)/2.0)
    sx, sy = s / aspect, s
    zz = (f+n)/(n-f)
    zw = 2*f*n/(n-f)
    return np.array([[sx,0,0,0],
                      [0,sy,0,0],
                      [0,0,zz,zw],
                      [0,0,-1,0]])

def frustum_projection(x0, x1, y0, y1, z0, z1):
    a = (x1+x0)/(x1-x0)
    b = (y1+y0)/(y1-y0)
    c = -(z1+z0)/(z1-z0)
    d = -2*z1*z0/(z1-z0)
    sx = 2*z0/(x1-x0)
    sy = 2*z0/(y1-y0)
    return np.array([[sx, 0, a, 0],
                      [ 0,sy, b, 0],
                      [ 0, 0, c, d],
                      [ 0, 0,-1, 0]])

def opengl_project_from_opencv_project(opencv_project):
    """
    given a matrix K from opencv, returns the corresponding matrix K' for opengl
    """
    fx = opencv_project[0, 0]
    fy = opencv_project[1, 1]
    cx = opencv_project[0, 2]
    cy = opencv_project[1, 2]
    w = 512
    h = 512
    far = 100.0
    near = 0.1
    opengl_mtx = np.array([[2*fx/w, 0.0, (w - 2*cx)/w, 0.0],
                           [0.0, -2*fy/h, (h - 2*cy)/h, 0.0],
                           [0.0, 0.0, (-far - near) / (far - near), -2.0*far*near/(far-near)],
                           [0.0, 0.0, -1.0, 0.0]])
    return opengl_mtx

def opengl_coords_to_opencv_coords(opengl_transform):
    """
    converts coordinates of opengl to "vision" (opencv) coordinates by flipping y and z axes
    """
    assert opengl_transform.shape == (4, 4)
    transform = np.array([[1, 0, 0, 0],  # flip y and z
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    opencv_transform = np.matmul(opengl_transform, transform)
    return opencv_transform

def opencv_coords_to_opengl_coords(opencv_transform):
    """
    converts coordinates of "vision" (opencv) to opengl coordinates by flipping y and z axes
    """
    return opengl_coords_to_opencv_coords(opencv_transform)

def create_random_cameras_on_unit_sphere(n, r, device="cuda"):
    """
    creates a batch of world2view and view2camera transforms on a unit sphere looking at the center
    reminder: "view" is the camera frame of reference, and "camera" is the image frame of reference (openGL conventions)
    """
    locs = torch.randn((n, 3), device=device)
    locs = torch.nn.functional.normalize(locs, dim=1, eps=1e-6)
    locs = locs * r
    matrices = torch.empty((n, 4, 4), dtype=torch.float32, device=device)
    for i in range(len(locs)):
        matrices[i] = look_at_torch(locs[i],
                                       torch.zeros(3, dtype=torch.float32, device=device),
                                       torch.tensor([0.,1.,0.], device=device),
                                       device=device)
    v2w = matrices  # c2w
    w2v = torch.inverse(v2w)
    v2c = torch.tensor(perspective_projection(), dtype=torch.float32, device=device)
    return w2v, v2c

def to_np(arr: torch.Tensor):
    if type(arr) == torch.Tensor:
        return arr.detach().cpu().numpy()
    elif type(arr) == np.ndarray:
        return arr
    elif type(arr) == Image.Image:
        return np.array(arr)
    return arr.detach().cpu().numpy()

def to_torch(arr: np.array, dtype=None, device="cpu"):
    if dtype is None:
        return torch.tensor(arr, device=device)
    else:
        return torch.tensor(arr, dtype=dtype, device=device)

def to_8b(x: np.array, clip=True):
    """
    convert a numpy (float) array to 8 bit
    """
    if x.dtype == np.float32:
        if clip:
            x = np.clip(x, 0, 1)
        return (255 * x).astype(np.uint8)
    elif x.dtype == np.uint8:
        return x

def to_float(x: np.array, clip=True):
    """
    convert a numpy (8bit) array to float
    """
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255
    elif x.dtype == np.float32:
        if clip:
            x = np.clip(x, 0, 1)
        return x

def to_PIL(x: np.array):
    """
    convert a numpy float array to a PIL image
    """
    if x.ndim == 3:
        return Image.fromarray(to_8b(x))
    elif x.ndim == 2:
        return Image.fromarray(to_8b(x[:, :, None]), mode="L")

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
    if s.shape[-1] != 3:
        raise ValueError("translation vector must be 3d")
    if s.ndim == 1:
        s = s[None, :]
    mat = np.diag(s)
    return to_44(mat)

def sincos(a):
    a = np.deg2rad(a)
    return np.sin(a), np.cos(a)

def rotate(a, r):
    """
    creates a rotation matrix from an angle and an axis
    """
    s, c = sincos(a)
    r = normalize(r)
    nc = 1 - c
    x, y, z = r
    return np.array([[x*x*nc +   c, x*y*nc - z*s, x*z*nc + y*s, 0],
                      [y*x*nc + z*s, y*y*nc +   c, y*z*nc - x*s, 0],
                      [x*z*nc - y*s, y*z*nc + x*s, z*z*nc +   c, 0],
                      [           0,            0,            0, 1]])

def rotx(a):
    """
    creates a rotation matrix around the x axis
    """
    s, c = sincos(a)
    return np.array([[1,0,0,0],
                      [0,c,-s,0],
                      [0,s,c,0],
                      [0,0,0,1]])

def roty(a):
    """
    creates a rotation matrix around the y axis
    """
    s, c = sincos(a)
    return np.array([[c,0,s,0],
                      [0,1,0,0],
                      [-s,0,c,0],
                      [0,0,0,1]])

def rotz(a):
    """
    creates a rotation matrix around the z axis
    """
    s, c = sincos(a)
    return np.array([[c,-s,0,0],
                      [s,c,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])