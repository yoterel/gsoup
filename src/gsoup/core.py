import torch
import numpy as np
from PIL import Image

def to_hom(x):
    """
    converts a vector to homogeneous coordinates
    :param x: nxc numpy array
    :return: nxc+1 numpy array
    """
    if type(x) == torch.Tensor:
        if x.ndim == 1:
            return torch.cat((x, torch.ones(1, device=x.device)))
        else:
            return torch.cat((x, torch.ones(x.shape[0], 1, device=x.device)), dim=-1)
    elif type(x) == np.ndarray:
        if x.ndim == 1:
            return np.concatenate((x, np.array([1], dtype=x.dtype)))
        else:
            return np.concatenate((x, np.ones((x.shape[0], 1), dtype=x.dtype)), axis=-1)
    else:
        raise ValueError("x must be torch.Tensor or np.ndarray")

def homogenize(x, keepdim=False):
    """
    normalizes a homogeneous vector by dividing by the last coordinate
    :param x: nx3 numpy array
    :return: nx4 numpy array
    """
    x = (x / x[..., -1:])
    if not keepdim:
        x = x[..., :-1]
    return x

def normalize(x, eps=1e-7):
    if type(x) == torch.Tensor:
        return x / (torch.norm(x, dim=-1, keepdim=True) + eps)
    elif type(x) == np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def broadcast_batch(*args):
    """
    broadcast a list of arrays to the same shape on the batch dimension
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

def to_34(mat: np.array):
    """
    converts a 4x4 to a 3x4 matrix by removeing the last row
    :param mat: 4x4 numpy array
    :return: 3x4 numpy array
    """
    if mat.ndim == 3:
        if mat.shape[1:] != (4, 4):
            raise ValueError("mat must be 4x4")
        return mat[:, :-1, :]
    else:
        if mat.shape != (4, 4):
            raise ValueError("mat must be 4x4")
        return mat[:-1, :]

def look_at_np(from_, to_, up_, openGL=False):
    """
    returns a batch of look_at transforms 4x4 (camera->world, the inverse of a ModelView matrix)
    will broadcast upon batch dimension if necessary.
    :param from_: n x 3 from vectors in world space
    :param to_: n x 3 at vector in world space
    :param up_: n x 3 up vector in world space
    :param is_openGL: if True, output will be in opengl coordinates (z backward, y up) otherwise (z forward, y down)
    :return: n x 4 x 4 transformation matrices (camera->world, the inverse of a ModelView matrix)
    """
    from_, to_, up_ = broadcast_batch(from_, to_, up_)
    forward = to_ - from_
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
    right = np.cross(up_, forward)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)
    rot = np.concatenate((right[..., None], up[..., None], forward[..., None]), axis=-1)
    c2w = np.concatenate((rot, from_[..., None]), axis=-1)
    c2w = to_44(c2w)
    if openGL:
        c2w[:, :, 1] *= -1
        c2w[:, :, 2] *= -1
    return c2w

def look_at_torch(
        eye:torch.Tensor, #3
        at:torch.Tensor, #3
        up:torch.Tensor, #3
        device:torch.device
    ) -> torch.Tensor: #4,4
    """
    creates a lookat transform matrix (OpenCV convention)
    :param eye: where the camera is
    :param at: where the camera is looking
    :param up: the up vector
    :param device: the device to put the matrix on
    :return: 4x4 lookat transform matrix
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
    """
    creates an orthographic projection matrix (OpenGL convention)
    :param l: left
    :param r: right
    :param b: bottom
    :param t: top
    :param n: near
    :param f: far
    :return: 4x4 orthographic projection matrix
    """
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
    """
    creates a perspective projection matrix (OpenGL convention)
    :param fovy: field of view in y direction
    :param aspect: aspect ratio
    :param n: near plane
    :param f: far plane
    :return: 4x4 projection matrix
    """
    s = 1.0/np.tan(np.deg2rad(fovy)/2.0)
    sx, sy = s / aspect, s
    zz = (f+n)/(n-f)
    zw = 2*f*n/(n-f)
    return np.array([[sx,0,0,0],
                      [0,sy,0,0],
                      [0,0,zz,zw],
                      [0,0,-1,0]])

def frustum_projection(x0, x1, y0, y1, z0, z1):
    """
    creates a projection matrix from a frustum (OpenGL convention)
    :param x0: left
    :param x1: right
    :param y0: bottom
    :param y1: top
    :param z0: near
    :param z1: far
    :return: 4x4 projection matrix
    """
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

def opengl_project_from_opencv_intrinsics(opencv_intrinsics, width, height, near=0.1, far=100.0):
    """
    given a matrix K from opencv, returns the corresponding projection matrix for opengl ("Eye/Camera/View space -> Clip Space")
    :param opencv_project: 3x3 projection matrix from opencv
    :param width: width of the image
    :param height: height of the image
    :param near: near plane
    :param far: far plane
    :return: 4x4 projection matrix for opengl (note: column major)
    """
    fx = opencv_intrinsics[0, 0]
    fy = opencv_intrinsics[1, 1]
    cx = opencv_intrinsics[0, 2]
    cy = opencv_intrinsics[1, 2]
    opengl_mtx = np.array([[2*fx/width, 0.0, (width - 2*cx)/width, 0.0],
                           [0.0, -2*fy/height, (height - 2*cy)/height, 0.0],
                           [0.0, 0.0, (-far - near) / (far - near), -2.0*far*near/(far-near)],
                           [0.0, 0.0, -1.0, 0.0]])
    return opengl_mtx

def opengl_c2w_to_opencv_c2w(opengl_transforms):
    """
    given a modelview matrix (World space->Eye/Camera/View space) where z is backward and y is up,
    converts its coordinate system to opencv convention (z forward, y down)
    :param opengl_transforms: 4x4 modelview matrix or batch of 4x4 modelview matrices
    :return: 4x4 modelview matrix in opencv convention
    """
    if opengl_transforms.ndim == 2:
        my_transforms = opengl_transforms[None, ...]
    else:
        my_transforms = opengl_transforms
    if my_transforms.shape[1:] != (4, 4):
        raise ValueError("transform must be 4x4 or batch of 4x4")
    my_transforms[:, :, 1] *= -1
    my_transforms[:, :, 2] *= -1
    return my_transforms.reshape(opengl_transforms.shape)

def opencv_c2w_to_opengl_c2w(opencv_transform):
    """
    converts coordinates of "vision" (opencv) to opengl coordinates by flipping y and z axes
    """
    return opengl_c2w_to_opencv_c2w(opencv_transform)

def create_random_cameras_on_unit_sphere(n, r, device="cuda"):
    """
    creates a batch of world2view ("ModelView" matrix) and view2clip ("Projection" matrix) transforms on a unit sphere looking at the center
    :param n: number of cameras
    :param r: radius of the sphere
    :param device: device to put the tensors on
    :return: world2view, view2clip
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
    """
    converts a tensor to numpy array
    :param arr: tensor
    :return: numpy array
    """
    if type(arr) == torch.Tensor:
        return arr.detach().cpu().numpy()
    elif type(arr) == np.ndarray:
        return arr
    elif type(arr) == Image.Image:
        return np.array(arr)
    return arr.detach().cpu().numpy()

def to_torch(arr: np.array, device="cpu", dtype=None):
    """
    converts a numpy array to a torch tensor
    :param arr: numpy array
    :param dtype: dtype of the tensor
    :param device: device to put the tensor on
    :return: torch tensor
    """
    if dtype is None:
        return torch.tensor(arr, device=device)
    else:
        return torch.tensor(arr, dtype=dtype, device=device)

def to_8b(x: np.array, clip=True):
    """
    convert a numpy (float, double) array to 8 bit
    """
    if x.dtype == np.float32 or x.dtype == np.float64:
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
    else:
        raise ValueError("unsupported dtype")

def to_PIL(x: np.array):
    """
    convert a numpy float array to a PIL image
    """
    if x.ndim == 3:
        return Image.fromarray(to_8b(x))
    elif x.ndim == 2:
        return Image.fromarray(to_8b(x[:, :, None]), mode="L")
    else:
        raise ValueError("unsupported array dimensions")

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