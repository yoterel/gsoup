import torch
import numpy as np


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
    :param from_: n x 3 from vectors in world space
    :param to_: 1 x 3 at vector in world space
    :param up_: 1 x 3 up vector in world space
    :return: n x 4 x 4 transformation matrices (c2w)
    """
    forward = to_[None, :] - from_
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
    right = np.cross(up_[None, :], forward)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)
    rot = np.concatenate((right[:, :, None], up[:, :, None], forward[:, :, None]), axis=-1)
    c2w = np.concatenate((rot, from_[:, :, None]), axis=-1)
    return to_44(c2w)

def opengl_to_opencv(opengl_transform):
    """
    converts a standard opengl transform to opencv transform by flipping y and z axes
    """
    assert opengl_transform.shape == (4, 4)
    transform = np.array([[1, 0, 0, 0],  # flip y and z
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    opencv_transform = np.matmul(opengl_transform, transform)
    return opencv_transform

def to_np(arr: torch.Tensor):
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
    if clip:
        x = np.clip(x, 0, 1)
    return (255 * x).astype(np.uint8)

def to_float(x: np.array):
    """
    convert a numpy (8bit) array to float
    """
    return (x.astype(np.float32) / 255)

