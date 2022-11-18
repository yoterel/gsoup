import torch
import numpy as np

def get_aspect_ratio(v, f):
    """
    measure aspect_ratio by circumradius / 2inradius
    aspect_ratio < 1 for illegal or degenerate triangle
    aspect_ratio = 1 for equilateral triangle
    aspect_ratio > 1 everything else
    v: vertices (V,3)
    f: faces (F,3)
    """
    v0, v1, v2 = v[f].unbind(dim=-2)
    a = (v0 - v1).norm(dim=-1)
    b = (v0 - v2).norm(dim=-1)
    c = (v1 - v2).norm(dim=-1)
    s = (a + b + c) / 2
    return a*b*c/(8*(s-a)*(s-b)*(s-c))

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

def calc_face_normals(vertices: torch.Tensor, faces: torch.Tensor, normalize: bool = False):
    """
    # V,3 first vertex may be unreferenced
    # F,3 long, first face may be all zeros

         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces]  # F,(3,3)
    v0, v1, v2 = full_vertices.unbind(dim=1)  # F,3
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # F,3
    if normalize:
        face_normals = torch.nn.functional.normalize(face_normals, eps=1e-6, dim=1)  # TODO inplace?
    return face_normals  # F,3

def calc_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor, face_normals: torch.Tensor = None):
    """
    # V,3 first vertex may be unreferenced
    # F,3 long, first face may be all zero
    return # F,3, not normalized
    """
    F = faces.shape[0]
    if face_normals is None:
        face_normals = calc_face_normals(vertices, faces)
    vertex_normals = torch.zeros((vertices.shape[0], 3, 3), dtype=vertices.dtype, device=vertices.device)  # V,C=3,3
    vertex_normals.scatter_add_(dim=0, index=faces[:, :, None].expand(F, 3, 3),
                                src=face_normals[:, None, :].expand(F, 3, 3))
    vertex_normals = vertex_normals.sum(dim=1)  # V,3
    return torch.nn.functional.normalize(vertex_normals, eps=1e-6, dim=1)

def calc_edges(faces: torch.Tensor, with_edge_to_face: bool = False, with_dummies=True):
    """
    # F,3 long - first face may be dummy with all zeros
    returns tuple of
    - edges E,2 long, 0 for unused, lower vertex index first
    - face_to_edge F,3 long
    - (optional) edge_to_face shape=E,[left,right],[face,side]

    o-<-----e1     e0,e1...edge, e0<e1
    |      /A      L,R....left and right face
    |  L /  |      both triangles ordered counter clockwise
    |  / R  |      normals pointing out of screen
    V/      |
    e0---->-o
    """

    F = faces.shape[0]
    # make full edges, lower vertex index first
    face_edges = torch.stack((faces, faces.roll(-1, 1)), dim=-1)  # F*3,3,2
    full_edges = face_edges.reshape(F * 3, 2)
    sorted_edges, _ = full_edges.sort(dim=-1)  # F*3,2 TODO min/max faster?

    # make unique edges
    edges, full_to_unique = torch.unique(input=sorted_edges, return_inverse=True, dim=0)  # (E,2),(F*3)
    E = edges.shape[0]
    face_to_edge = full_to_unique.reshape(F, 3)  # F,3
    if not with_edge_to_face:
        return edges, face_to_edge

    is_right = full_edges[:, 0] != sorted_edges[:, 0]  # F*3
    edge_to_face = torch.zeros((E, 2, 2), dtype=torch.long, device=faces.device)  # E,LR=2,S=2
    scatter_src = torch.cartesian_prod(torch.arange(0, F, device=faces.device),
                                       torch.arange(0, 3, device=faces.device))  # F*3,2
    edge_to_face.reshape(2 * E, 2).scatter_(dim=0, index=(2 * full_to_unique + is_right)[:, None].expand(F * 3, 2),
                                            src=scatter_src)  # E,LR=2,S=2
    if with_dummies:
        edge_to_face[0] = 0
    return edges, face_to_edge, edge_to_face

def calc_edge_length(vertices: torch.Tensor, edges: torch.Tensor):
    """
    # V,3 first may be dummy
    return # E,2 long, lower vertex index first, (0,0) for unused
    """
    full_vertices = vertices[edges]  # E,2,3
    a, b = full_vertices.unbind(dim=1)  # E,3
    return torch.norm(a - b, p=2, dim=-1)

def prepend_dummies(vertices: torch.Tensor, faces: torch.Tensor):
    """
    prepend dummy elements to vertices and faces to enable "masked" scatter operations
    # V,D
    # F,3 long
    """
    V, D = vertices.shape
    vertices = torch.concat((torch.full((1, D), fill_value=torch.nan, device=vertices.device), vertices), dim=0)
    faces = torch.concat((torch.zeros((1, 3), dtype=torch.long, device=faces.device), faces + 1), dim=0)
    return vertices, faces


def remove_dummies(vertices: torch.Tensor, faces: torch.Tensor):
    """
    remove dummy elements added with prepend_dummies()
    # V,D - first vertex all nan and unreferenced
    # F,3 long - first face all zeros
    """
    return vertices[1:], faces[1:] - 1