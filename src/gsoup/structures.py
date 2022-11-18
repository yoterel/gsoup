import numpy as np

def get_gizmo_coords(scale=20.0):
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]).astype(np.float32)
    vertices *= scale
    return vertices

def get_camera_coords(scale=0.1):
    vertices = np.array([
        # origin
        [0, 0, 0],
        # frustrum
        [0.5, 0.5, 1],
        [-0.5, 0.5, 1],
        [-0.5, -0.5, 1],
        [0.5, -0.5, 1],
        # axis
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    vertices *= scale
    edges = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [0, 5], [0, 6], [0, 7],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ])
    colors = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],

        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],

        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])
    return vertices, edges, colors

def get_aabb_coords(centered=True):
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]).astype(np.float32)
    if centered:
        vertices -= np.array([0.5, 0.5, 0.5])
    edges = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [4, 1],
        [4, 2],
        [4, 7],
        [6, 1],
        [6, 7],
        [6, 3],
        [5, 2],
        [5, 7],
        [5, 3],

        ])
    colors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],

        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])
    return vertices, edges, colors