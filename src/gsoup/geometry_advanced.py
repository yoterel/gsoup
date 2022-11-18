import torch
import numpy as np


def merge_meshes(v1, f1, v2, f2):
    """merge two meshes into one"""
    v = np.concatenate([v1, v2], axis=0)
    f = np.concatenate([f1, f2 + v1.shape[0]], axis=0)
    return v, f