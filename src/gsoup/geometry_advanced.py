import torch
import numpy as np
from .core import to_hom
from .geometry_basic import get_edges, edge_contraction


def distribute_field(v, f, field, avg=False):
    """
    given a field of size F (with any number of addtional dimensions), distribute it to the vertices of the mesh by summing up the values of the incident faces
    :param v: vertices of the mesh
    :param f: faces of the mesh
    :param field: field of size F
    :param avg: if True, the field is averaged over the incident faces instead of summed up
    :return: scalar field of size V
    """
    # compute the incident triangles per vertex
    incident_triangles = compute_incident_triangles(f)
    # compute the scalar field per vertex
    field_per_vertex = np.zeros((v.shape[0], *field.shape[1:]))
    for i, face in enumerate(f):
        for vertex in face:
            field_per_vertex[vertex] += field[i]
    if avg:
        field_per_vertex /= len(incident_triangles)
    return field_per_vertex


def distribute_scalar_field(num_vertices, f, per_face_scalar_field, avg=False):
    """
    computes a scalar field per vertex by summing / averaging the incident face scalar field
    :param num_vertices: number of vertices in the mesh
    :param f: faces of the mesh
    :param per_face_scalar_field: scalar field of size F
    :param avg: if True, the field is averaged over the incident faces instead of summed up
    :return: scalar field of size V
    """
    device = f.device
    incident_face_areas = torch.zeros([num_vertices, 1], device=device)
    f_unrolled = f.flatten()
    face_indices_repeated_per_vertex = torch.arange(f.shape[0], device=f.device)
    face_indices_repeated_per_vertex = torch.repeat_interleave(face_indices_repeated_per_vertex, repeats=3)
    face_areas_repeated_per_face = per_face_scalar_field[face_indices_repeated_per_vertex].unsqueeze(-1)
    incident_face_areas = torch.index_add(incident_face_areas, dim=0, index=f_unrolled,
                                          source=face_areas_repeated_per_face)
    if avg:
        neighbors = torch.index_add(torch.zeros_like(incident_face_areas), dim=0, index=f_unrolled,
                                    source=torch.ones_like(face_areas_repeated_per_face))
        incident_face_areas = incident_face_areas / neighbors
    return incident_face_areas


def distribute_vector_field(num_vertices, f, per_face_vector_field):
    """
    computes a vector field per vertex by summing the incident face vector field
    :param num_vertices: number of vertices in the mesh
    :param f: faces of the mesh
    :param per_face_vector_field: vector field of size Fx3
    :return: vector field of size Vx3
    """
    device = f.device
    incident_face_areas = torch.zeros([num_vertices, 1], device=device)
    f_unrolled = f.flatten()
    face_indices_repeated_per_vertex = torch.arange(f.shape[0], device=f.device)
    face_indices_repeated_per_vertex = torch.repeat_interleave(face_indices_repeated_per_vertex, repeats=3)
    normals_repeated_per_face = per_face_vector_field[face_indices_repeated_per_vertex]
    normals_repeated_per_face = normals_repeated_per_face
    incident_face_vectors = torch.zeros([num_vertices,per_face_vector_field.shape[-1]], device=device)
    incident_face_vectors = torch.index_add(incident_face_vectors, dim=0, index=f_unrolled,
                                            source=normals_repeated_per_face)
    return incident_face_vectors


def compute_incident_triangles(f):
    """
    compute the incident triangles per vertex
    :param v: vertices of the mesh
    :param f: faces of the mesh
    :return: list of lists of incident triangles per vertex
    """
    incident_triangles = [[] for _ in range(np.unique(f).shape[0])]
    for i, face in enumerate(f):
        for vertex in face:
            incident_triangles[vertex].append(i)
    return incident_triangles


def compute_quadtratic_surface(v, f):
    """
    given a mesh, compute the quadratic surface coefficients per vertex in a tensor of size Vx4x4
    the per vertex quadratic surface is computed from the sum of quadtratic surfaces of incident faces to each vertex
    :param v: vertices of the mesh
    :param f: faces of the mesh
    :return: tensor of size Vx4x4 of quadtratic surface coefficients per vertex
    """
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    e1 = e1 / np.linalg.norm(e1, axis=-1)[: ,None]
    e2 = e2 / np.linalg.norm(e2, axis=-1)[: ,None]
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n, axis=-1)[: ,None]
    d = ((-v[f[:, 0]])[:, None, :] @ n[:, :, None]).squeeze()
    Qf = np.zeros((f.shape[0], 4, 4))
    Qf[:, :3, :3] = n[:, :, None] @ n[:, None, :]
    Qf[:, -1, :-1] = d[:, None] * n
    Qf[:, :-1, -1] = Qf[:, -1, :-1]
    Qf[:, -1, -1] = d * d
    Qv = distribute_field(v, f, Qf)
    return Qv


def compute_costs(v_hats, valid_pairs, Qv):
    """
    compute the costs of contracting the valid pairs
    :param v_hats: the new vertices after contraction
    :param valid_pairs: the valid pairs of vertices to contract
    :param Qv: the quadratic surface coefficients per v_hat
    :return: the costs of contracting the valid pairs
    """
    v_hats = to_hom(v_hats)
    Q_hats = np.sum(Qv[valid_pairs], axis=1)
    return (v_hats[:, None, :] @ (Q_hats @ v_hats[:, :, None])).squeeze()


def compute_v_hats(v, valid_pairs, Qv):
    """
    computes the optimal vertex positions for each valid pair of vertices given the quadratic surface coefficients of contracting the pairs
    :param v: vertices of the mesh
    :param valid_pairs: list of valid pairs of vertices
    :param Qv: tensor of size Vx4x4 of quadtratic surface coefficients per vertex
    :return: tensor of size Vx3 of (almost) optimal vertex positions
    """
    v_hats = np.zeros((valid_pairs.shape[0], 3), dtype=np.float32)
    Q_hats = np.sum(Qv[valid_pairs], axis=1)
    Q_hats[:, -1, :] = np.array([0, 0, 0, 1])[None, :]
    mask = np.linalg.det(Q_hats) != 0
    v_hats[mask] = (np.linalg.inv(Q_hats[mask]) @ np.array([0, 0, 0, 1])[None, :, None]).squeeze()[:, :3]  # d(cost)/d(v_hat) = 0
    # note: orig paper searches for the minimum along the segment between the two vertices if Q_hat is singular
    v_hats[~mask] = np.mean(v[valid_pairs[~mask]], axis=1)  # if Q_hats is singular, use the mean of the two vertices
    return v_hats

def qslim(v: np.ndarray, f: np.ndarray, budget: int):
    """
    A slightly naive implementation of "Surface Simplification Using Quadric Error Metrics", 1997
    """
    if v.ndim != 2 or v.shape[-1] != 3:
        raise ValueError("v must be V x 3 array")
    if f.ndim != 2 or f.shape[-1] != 3:
        raise ValueError("f must be F x 3 array")
    if budget < 4:
        raise ValueError("minimum budget is 3 triangles")
    while f.shape[0] > budget:
        print("current # faces: {}".format(f.shape[0]))
        valid_pairs = get_edges(f)
        Qv = compute_quadtratic_surface(v, f)
        v_hats = compute_v_hats(v, valid_pairs, Qv)
        costs = compute_costs(v_hats, valid_pairs, Qv)
        lowest_cost = np.argmin(costs)
        v, f = edge_contraction(v, f, valid_pairs[lowest_cost], v_hats[lowest_cost])
    return v, f