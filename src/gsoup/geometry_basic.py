import torch
import numpy as np

def normalize_vertices(vertices):
    eps = 1e-7
    """shift and resize mesh to fit into a unit sphere"""
    if type(vertices) == np.ndarray:
        vertices -= (vertices.min(axis=0)[0] + vertices.max(axis=0)[0]) / 2
        vertices = vertices / (np.linalg.norm(vertices, axis=-1).max() + eps)
    elif type(vertices) == torch.Tensor:
        vertices -= (vertices.min(dim=0)[0] + vertices.max(dim=0)[0]) / 2
        vertices = vertices / (torch.norm(vertices, dim=-1).max() + eps)
    else:
        raise TypeError("vertices must be either np.ndarray or torch.Tensor")
    return vertices