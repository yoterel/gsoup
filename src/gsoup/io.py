import torch
import numpy as np
from pathlib import Path
import igl


def load_obj(path: Path, load_normals=False, to_torch=False, device=None):
    """
    needs explaining?
    use igl backend.
    """
    v, _, n, f, _, _ = igl.read_obj(str(path))
    if to_torch and device is not None:
        v = torch.tensor(v, dtype=torch.float, device=device)
        f = torch.tensor(f,dtype=torch.long, device=device)
        n = torch.tensor(n, dtype=torch.float, device=device)
    if load_normals:
        return v, f, n
    else:
        return v, f

def save_obj(path: Path, vertices, faces):
    """"
    saves a mesh as an obj file
    use igl backend.
    """
    filename = Path(filename)
    if filename.suffix not in [".obj", ".ply"]:
        raise ValueError("Only .obj and .ply are supported")
    if type(vertices) == torch.Tensor:
        vertices = vertices.detach().cpu().numpy()
    if type(faces) == torch.Tensor:
        faces = faces.detach().cpu().numpy()
    if (faces < 0).any():
        raise ValueError("Faces must be positive")
    if np.isnan(vertices).any():
        raise ValueError("Vertices must be finite")
    if np.isnan(faces).any():
        raise ValueError("Faces must be finite")
    if vertices.dtype != np.float32:
        raise ValueError("Vertices must be of type float32")
    if faces.dtype != np.int64:
        raise ValueError("Faces must be of type int64")
    igl.write_obj(str(path), vertices, faces)