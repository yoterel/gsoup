import torch
import numpy as np
from PIL import Image


def is_np(x):
    """Checks if x is a numpy array or torch tensor.
    
    Args:
        x: Object to check.
        
    Returns:
        bool: True if x is a numpy array, False if x is a torch tensor.
        
    Raises:
        ValueError: If x is neither a numpy array nor a torch tensor.
    """
    if type(x) == np.ndarray:
        return True
    elif type(x) == torch.Tensor:
        return False
    else:
        raise ValueError("input must be torch.Tensor or np.ndarray")


def is_float(x):
    """Checks if x is a float array.
    
    Args:
        x: Object to check.
        
    Returns:
        bool: True if x is a float array, False if x is not.
    """
    if is_np(x):
        return np.issubdtype(x.dtype, np.floating)
    else:
        return torch.is_floating_point(x)


def permute_channel_dimension(x):
    """Permutes the channels of a numpy array or torch tensor.
    
    Args:
        x: Numpy array or torch tensor of shape (h, w, c), (c, h, w), (b, h, w, c), or (b, c, h, w).
        
    Returns:
        Array or tensor with permuted channels.
        
    Raises:
        ValueError: If x is not 3 or 4 dimensional.
    """
    if not 2 < x.ndim <= 4:
        raise ValueError("x must be 3 or 4 dimensional")
    if is_np(x):
        if x.ndim == 3:
            x = x.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)
        elif x.ndim == 4:
            x = x.transpose(0, 3, 1, 2)  # (b, h, w, c) -> (b, c, h, w)
    else:
        if x.ndim == 3:
            x = x.permute(1, 2, 0)  # (c, h, w) -> (h, w, c)
        elif x.ndim == 4:
            x = x.permute(0, 2, 3, 1)  # (b, c, h, w) -> (b, h, w, c)
    return x


def to_hom(x):
    """Converts a vector to homogeneous coordinates.
    
    Concatenates 1 along the last dimension.
    
    Args:
        x: Numpy array or torch tensor of shape (..., c).
        
    Returns:
        Numpy array or torch tensor of shape (..., c+1).
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
    """Normalizes a homogeneous vector by dividing by the last coordinate.
    
    Args:
        x: Homogeneous vector array.
        keepdim: If True, keeps the homogeneous dimension. Defaults to False.
        
    Returns:
        Normalized vector array.
    """
    x = x / x[..., -1:]
    if not keepdim:
        x = x[..., :-1]
    return x


def normalize(x, eps=1e-7):
    """Normalizes a vector by dividing by its norm.
    
    Args:
        x: Numpy array or torch tensor of shape (..., c).
        eps: Small epsilon value to avoid division by zero. Defaults to 1e-7.
        
    Returns:
        Normalized array along the last dimension.
    """
    if is_np(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    else:
        return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def broadcast_batch(*args):
    """Broadcasts a list of arrays to the same shape on the batch dimension.
    
    Assumes first dimension is batch unless ndim = 1 for all inputs (but then does not broadcast).
    
    Args:
        *args: List of arrays to broadcast.
        
    Returns:
        List of arrays with the same shape.
        
    Raises:
        ValueError: If cannot broadcast 1d and nd arrays.
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


def repeat(x, n):
    """Repeats a numpy array or torch tensor.
    
    Args:
        x: The input array or tensor.
        n: A tuple describing the repeat per dimension of x.
        
    Returns:
        The repeated input.
    """
    if is_np(x):
        y = np.tile(x, n)
    else:
        y = torch.repeat(x, *n)
    return y


def compose_rt(R: np.array, t: np.array, square=False):
    """Composes a rotation-translation matrix from rotation and translation.
    
    Will broadcast upon batch dimension if necessary.
    
    Args:
        R: Rotation matrix of shape (n, 3, 3).
        t: Translation vector of shape (n, 3).
        square: If True, output will be (n, 4, 4), otherwise (n, 3, 4). Defaults to False.
        
    Returns:
        Composition of the rotation and translation.
    """
    RR, tt = broadcast_batch(R, t)
    Rt = np.concatenate((RR, tt[:, :, None]), axis=-1)
    if square:
        Rt = to_44(Rt)
    return Rt


def to_44(mat):
    """Converts a (3,4) to a (4,4) matrix by concatenating [0, 0, 0, 1].
    
    Args:
        mat: Matrix of shape (n, 3, 4) where n can be any number including 0.
        
    Returns:
        Matrix of shape (n, 4, 4).
        
    Raises:
        ValueError: If mat is not (..., 3, 4) or (..., 4, 4).
    """
    if mat.shape[-2:] == (4, 4):
        return mat
    if mat.shape[-2:] != (3, 4):
        raise ValueError("mat must be 3x4")
    if is_np(mat):
        to_cat = np.broadcast_to(
            np.array([0, 0, 0, 1]).astype(mat.dtype), (*mat.shape[:-2], 1, 4)
        )
        new_mat = np.concatenate((mat, to_cat), axis=-2)
    else:
        to_cat = torch.zeros(
            (*mat.shape[:-2], 1, 4), dtype=mat.dtype, device=mat.device
        )
        to_cat[..., -1] = 1
        new_mat = torch.cat((mat, to_cat), dim=-2)
    return new_mat


def to_34(mat: np.array):
    """Converts a (n, 4, 4) to a (n, 3, 4) matrix by removing the last row.
    
    Args:
        mat: Matrix of shape (n, 4, 4).
        
    Returns:
        Matrix of shape (n, 3, 4).
        
    Raises:
        ValueError: If mat is not (..., 4, 4).
    """
    if mat.ndim == 3:
        if mat.shape[1:] != (4, 4):
            raise ValueError("mat must be 4x4")
        return mat[:, :-1, :]
    else:
        if mat.shape != (4, 4):
            raise ValueError("mat must be 4x4")
        return mat[:-1, :]


def to_np(x, permute_channels=False):
    """Converts input to numpy array.
    
    Args:
        x: Input tensor, array, PIL Image, or list.
        permute_channels: If True and input is a torch tensor, permutes the channels order to channels last. Defaults to False.
        
    Returns:
        Numpy array.
        
    Raises:
        TypeError: If input type cannot be converted to numpy array.
    """
    if type(x) == torch.Tensor:
        if permute_channels:
            x = permute_channel_dimension(x)
        return x.detach().cpu().numpy()
    elif type(x) == np.ndarray:
        return x
    elif type(x) == Image.Image:
        return np.array(x)
    elif type(x) == list:
        return np.array(x)
    else:
        raise TypeError("cannot convert {} to numpy array".format(str(type(x))))


def to_numpy(x, permute_channels=False):
    """Converts input to numpy array.
    
    See to_np for detailed documentation.
    
    Args:
        x: Input tensor, array, PIL Image, or list.
        permute_channels: If True and input is a torch tensor, permutes the channels order to channels last. Defaults to False.
        
    Returns:
        Numpy array.
    """
    return to_np(x, permute_channels)


def to_torch(x, device="cpu", dtype=None, permute_channels=False):
    """Converts a numpy array to a torch tensor.
    
    Args:
        x: Numpy array to convert.
        device: Device to put the tensor on. Defaults to "cpu".
        dtype: Dtype of the tensor. If None, will be inferred from the input. Defaults to None.
        permute_channels: If True, permutes the channels order. Defaults to False.
        
    Returns:
        Torch tensor.
    """
    if is_np(x):
        if permute_channels:
            x = permute_channel_dimension(x)
        if dtype is None:
            x = torch.tensor(x, device=device)
        else:
            x = torch.tensor(x, dtype=dtype, device=device)
        return x
    else:
        return x


def to_8b(x, clip=True):
    """Converts an array to 8-bit format.
    
    Args:
        x: Input array (float, double, bool, or uint8).
        clip: If True, clips values to [0,1]. Defaults to True.
        
    Returns:
        8-bit array.
        
    Raises:
        ValueError: If unsupported dtype.
    """
    if is_np(x):
        if is_float(x):
            if clip:
                x = np.clip(x, 0, 1)
            return (255 * x).round().astype(np.uint8)
        elif x.dtype == bool:
            return x.astype(np.uint8) * 255
        elif x.dtype == np.uint8:
            return x
        else:
            raise ValueError("unsupported dtype")
    else:
        if is_float(x):
            if clip:
                x = torch.clamp(x, 0, 1)
            return (255 * x).round().type(torch.uint8)
        elif x.dtype == torch.bool:
            return x.type(torch.uint8) * 255
        elif x.dtype == torch.uint8:
            return x
        else:
            raise ValueError("unsupported dtype")


def to_float(x, clip=True):
    """Converts an 8-bit or bool array to float.
    
    Args:
        x: Input array (uint8, bool, or float).
        clip: If True, clips values to [0,1]. Defaults to True.
        
    Returns:
        Float array.
        
    Raises:
        ValueError: If unsupported dtype.
    """
    if is_np(x):
        if x.dtype == np.uint8:
            return x.astype(np.float32) / 255
        elif x.dtype == bool:
            return x.astype(np.float32)
        elif is_float(x):
            if clip:
                x = np.clip(x, 0, 1)
            return x
        else:
            raise ValueError("unsupported dtype")
    else:
        if x.dtype == torch.uint8:
            return x.to(torch.float32) / 255
        elif x.dtype == torch.bool:
            return x.to(torch.float32)
        elif is_float(x):
            if clip:
                x = torch.clamp(x, 0, 1)
            return x
        else:
            raise ValueError("unsupported dtype")


def to_PIL(x):
    """Converts a numpy array or torch tensor to a PIL image.
    
    Args:
        x: Float numpy array or torch tensor of shape (h, w, 3) or (h, w).
        
    Returns:
        PIL image.
        
    Raises:
        ValueError: If unsupported array dimensions or dtype.
    """
    if is_float(x):
        if not is_np(x):
            x = to_np(x, permute_channels=True)
        if x.ndim == 3:
            return Image.fromarray(to_8b(x))
        elif x.ndim == 2:
            return Image.fromarray(to_8b(x[:, :, None]), mode="L")
        else:
            raise ValueError("unsupported array dimensions")
    else:
        raise ValueError("unsupported dtype")


def map_range(x, in_min, in_max, out_min, out_max):
    """Maps input from input range to output range.
    
    Args:
        x: Input array.
        in_min: Input range minimum.
        in_max: Input range maximum.
        out_min: Output range minimum.
        out_max: Output range maximum.
        
    Returns:
        Mapped input array.
    """
    x_01 = (x - in_min) / (in_max - in_min + 1e-6)
    return x_01 * (out_max - out_min) + out_min


def map_to_01(x, dims=None):
    """Maps an input to [0,1] range.
    
    Args:
        x: Input array.
        dims: The list of dimensions to map to [0,1]. Defaults to None, which maps the entire array.
        
    Returns:
        Mapped input array.
    """
    if is_np(x):
        return map_range(
            x,
            np.amin(x, axis=dims, keepdims=True),
            np.amax(x, axis=dims, keepdims=True),
            0.0,
            1.0,
        )
    else:
        return map_range(
            x, x.amin(dim=dims, keepdim=True), x.amax(dim=dims, keepdim=True), 0.0, 1.0
        )


def swap_columns(x, col1_index, col2_index):
    """Swaps two columns of a numpy array inplace.
    
    Args:
        x: Array of shape (..., c).
        col1_index: Index of the first column.
        col2_index: Index of the second column.
        
    Returns:
        Array with swapped columns.
    """
    x[..., [col2_index, col1_index]] = x[..., [col1_index, col2_index]]
    return x


def to_gray(x, keep_channels=False):
    """Converts image to grayscale.
    
    See color_to_gray for detailed documentation.
    
    Args:
        x: Input image array.
        keep_channels: If True, preserves channel dimension. Defaults to False.
        
    Returns:
        Grayscale version of the input.
    """
    return color_to_gray(x, keep_channels)


def rgb_to_gray(x, keep_channels=False):
    """Converts RGB image to grayscale.
    
    See color_to_gray for detailed documentation.
    
    Args:
        x: Input RGB image array.
        keep_channels: If True, preserves channel dimension. Defaults to False.
        
    Returns:
        Grayscale version of the input.
    """
    return color_to_gray(x, keep_channels)


def color_to_gray(x, keep_channels=False):
    """Converts image to grayscale by averaging over channels.
    
    Args:
        x: Numpy array or torch tensor of shape (n, h, w, c) or (h, w, c) where c >= 1.
        keep_channels: If True, will preserve c by repeating the array after average. Defaults to False.
        
    Returns:
        The grayscale version of x.
        
    Raises:
        ValueError: If ndim of x is not 3 (h, w, c) or 4 (n, h, w, c).
    """
    orig_ndim = x.ndim
    if (orig_ndim != 3) and (orig_ndim != 4):
        raise ValueError("ndim of x must be 3 (h, w, c) or 4 (n, h, w, c)")
    c = x.shape[-1]
    orig_dtype = x.dtype
    if is_np(x):
        if is_float(x):
            x = x.mean(axis=-1, keepdims=True)
        else:
            x = x.astype(np.float32).mean(axis=-1, keepdims=True)
            x = x.round().astype(orig_dtype)
        if keep_channels:
            if orig_ndim == 3:
                x = np.tile(x, (1, 1, c))
            else:
                x = np.tile(x, (1, 1, 1, c))
    else:
        if is_float(x):
            x = x.mean(dim=-1, keepdim=True)
        else:
            x = x.to(torch.float32).mean(dim=-1, keepdim=True)
            x = x.round().to(orig_dtype)
        if keep_channels:
            if orig_ndim == 3:
                x = x.repeat(1, 1, c)
            else:
                x = x.repeat(1, 1, 1, c)
    return x


def make_monotonic(y, increasing=True):
    """Ensures a 1D array is monotonic increasing or decreasing.
    
    Args:
        y: Numpy array of shape (N,) to be made monotonic.
        increasing: If True, enforce increasing monotonicity; if False, enforce decreasing. Defaults to True.
        
    Returns:
        Numpy array of the same shape as y, made monotonic.
    """
    if increasing:
        return np.maximum.accumulate(y)
    else:
        return np.minimum.accumulate(y)
