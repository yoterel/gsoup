import torch
import numpy as np
from PIL import Image


def is_np(x):
    """
    checks if x is a numpy array or torch tensor (will raise an error if x is neither)
    :param x: object to check
    :return: True if x is a numpy array, False if x is a torch tensor
    """
    if type(x) == np.ndarray:
        return True
    elif type(x) == torch.Tensor:
        return False
    else:
        raise ValueError("input must be torch.Tensor or np.ndarray")


def to_hom(x):
    """
    converts a vector to homogeneous coordinates (concatenates 1 along last dimension)
    :param x: (..., c) numpy array or torch tensor
    :return: (..., c+1) numpy array or torch tensor
    """
    if is_np(x):
        if x.ndim == 1:
            return np.concatenate((x, np.array([1], dtype=x.dtype)))
        else:
            return np.concatenate(
                (x, np.ones((*x.shape[:-1], 1), dtype=x.dtype)), axis=-1
            )
    else:
        if x.ndim == 1:
            return torch.cat((x, torch.ones(1, device=x.device)))
        else:
            return torch.cat((x, torch.ones(*x.shape[:-1], 1, device=x.device)), dim=-1)


def homogenize(x, keepdim=False):
    """
    normalizes a homogeneous vector by dividing by the last coordinate
    :param x: nx3 numpy array
    :return: nx4 numpy array
    """
    x = x / x[..., -1:]
    if not keepdim:
        x = x[..., :-1]
    return x


def normalize(x, eps=1e-7):
    """
    normalizes a vector by dividing by its norm
    :param x: (... , c) numpy array
    :return: (... , c) numpy array normalized along the last dimension
    """
    if is_np(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    else:
        return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


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


def compose_rt(R: np.array, t: np.array, square=False):
    """
    composes a n x 3 x 4 numpy array from rotation and translation.
    will broadcast upon batch dimension if necessary.
    :param R: nx3x3 numpy array
    :param t: nx3 numpy array
    :param square: if True, output will be 4x4, otherwise 3x4
    :return: n x 3 x 4 composition of the rotation and translation
    """
    RR, tt = broadcast_batch(R, t)
    Rt = np.concatenate((RR, tt[:, :, None]), axis=-1)
    if square:
        Rt = to_44(Rt)
    return Rt


def to_44(mat):
    """
    converts a 3x4 to a 4x4 matrix by concatenating 0 0 0 1
    :param mat: dimsx3x4 numpy array (dims can be any number of dims including 0)
    :return: dimsx4x4 numpy array
    """
    if mat.shape[-2:] == (4, 4):
        return mat
    if mat.shape[-2:] != (3, 4):
        raise ValueError("mat must be 3x4")
    if is_np(mat):
        to_cat = np.broadcast_to(np.array([0, 0, 0, 1]), (*mat.shape[:-2], 1, 4))
        new_mat = np.concatenate((mat, to_cat), axis=-2)
    else:
        to_cat = torch.zeros(
            (*mat.shape[:-2], 1, 4), dtype=mat.dtype, device=mat.device
        )
        to_cat[..., -1] = 1
        new_mat = torch.cat((mat, to_cat), dim=-2)
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


def to_numpy(arr: torch.Tensor):
    """
    converts a tensor to numpy array
    :param arr: tensor
    :return: numpy array
    """
    return to_np(arr)


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


def to_8b(x, clip=True):
    """
    convert an array (float, double) array to 8 bit
    :param x: array
    :param clip: if True, clips values to [0,1]
    :return: 8 bit array
    """
    if is_np(x):
        if x.dtype == np.float16 or x.dtype == np.float32 or x.dtype == np.float64:
            if clip:
                x = np.clip(x, 0, 1)
            return (255 * x).round().astype(np.uint8)
        elif x.dtype == bool:
            return x.astype(np.uint8) * 255
        elif x.dtype == np.uint8:
            return x
    else:
        if x.dtype == torch.float32 or x.dtype == torch.float64:
            if clip:
                x = torch.clamp(x, 0, 1)
            return (255 * x).round().type(torch.uint8)
        elif x.dtype == torch.bool:
            return x.type(torch.uint8) * 255
        elif x.dtype == torch.uint8:
            return x


def to_float(x, clip=True):
    """
    convert a 8bit or bool array to float
    :param x: array
    :param clip: if True, clips values to [0,1]
    :return: float array
    """
    if is_np(x):
        if x.dtype == np.uint8:
            return x.astype(np.float32) / 255
        elif x.dtype == np.float32:
            if clip:
                x = np.clip(x, 0, 1)
            return x
        elif x.dtype == bool:
            return x.astype(np.float32)
        else:
            raise ValueError("unsupported dtype")
    else:
        if x.dtype == torch.uint8:
            return x.to(torch.float32) / 255
        elif x.dtype == torch.float32:
            if clip:
                x = torch.clamp(x, 0, 1)
            return x
        elif x.dtype == torch.bool:
            return x.to(torch.float32)
        else:
            raise ValueError("unsupported dtype")


def to_PIL(x: np.array):
    """
    convert a numpy float array to a PIL image
    :param x: array
    :return: PIL image
    """
    if x.ndim == 3:
        return Image.fromarray(to_8b(x))
    elif x.ndim == 2:
        return Image.fromarray(to_8b(x[:, :, None]), mode="L")
    else:
        raise ValueError("unsupported array dimensions")


def map_range(x, out_min, out_max):
    """
    given an input and a range, maps it to a new range
    :param x: input
    :param out_min: output minimum
    :param out_max: output maximum
    :return: mapped input
    """
    if type(x) == np.ndarray:
        return np.clip(
            (x - x.min()) * (out_max - out_min) / (x.max() - x.min()) + out_min,
            out_min,
            out_max,
        )
    elif type(x) == torch.Tensor:
        return torch.clamp(
            (x - x.min()) * (out_max - out_min) / (x.max() - x.min()) + out_min,
            out_min,
            out_max,
        )
    else:
        raise ValueError("unsupported type")


def map_to_01(x):
    """
    maps an input to [0,1]
    :param x: input
    :return: mapped input
    """
    return map_range(x, 0, 1)


def swap_columns(x, col1_index, col2_index):
    """
    swaps two columns of a numpy array inplace
    :param x: array N x C
    :param col1_index: index of the first column
    :param col2_index: index of the second column
    :return: array with swapped columns
    """
    x[:, [col2_index, col1_index]] = x[:, [col1_index, col2_index]]
    return x
