import torch
import numpy as np
from .core import is_np, broadcast_batch


def scalar_projection(a, b):
    """
    computes a scalar projection of vector a onto b
    note: this is the size of the componenet of a tangent to b
    :param a, b: np arrays of size 2 or 3
    """
    return np.dot(a, b) / np.dot(b, b)


def project_point_to_line(p, a, b):
    """
    Returns the closest point on line ab to point p.
    p, a, b: numpy arrays of shape (2,).
    """
    if np.all(a == b):
        return a
    # direction vector of the line
    ab = b - a
    # direction of vector from a to p
    ap = p - a
    # compute the scalar projection of p-a onto b-a
    t = scalar_projection(ap, ab)
    # compute the projection point
    return a + t * ab


def project_point_to_segment(p, a, b):
    """
    Returns the closest point on a finite segment ab to point p.
    p, a, b: numpy arrays of shape (2,).
    """
    if np.all(a == b):
        return a
    # direction vector of the line
    ab = b - a
    # direction of vector from a to p
    ap = p - a
    # compute the scalar projection of p-a onto b-a
    t = scalar_projection(ap, ab)
    # clip result to either a or b
    t = np.clip(t, 0, 1)
    return a + t * ab


def point_line_distance(p, a, b):
    """
    returns the distance between a point p and a line defined by the points a and b
    :param p: point to check (3,) or batch of points to check (B, 3)
    :param a: first vertex of the line (3,)
    :param b: second vertex of the line (3,)
    :return the distance between the point and the line
    """
    nom = np.linalg.norm(np.cross(b - a, p - a), axis=-1)
    denom = np.linalg.norm(b - a, axis=-1)
    return nom / denom


def edge_function(a, b, p):
    """
    returns the "edge function" which equals half the area of the traingle formed by a, b and p
    if the result is positive, p is to the right of the line ab
    :param a: first vertex of the triangle (3,)
    :param b: second vertex of the triangle (3,)
    :param p: point to check (3,) or batch of points to check (B, 3)
    """
    return np.cross(b - a, p - a)


def is_inside_triangle(p, a, b, c):
    """
    checks if a point is inside a triangle
    :param p: point or batch of points to check (3,)
    :param a: first vertex of the triangle (3,)
    :param b: second vertex of the triangle (3,)
    :param c: third vertex of the triangle (3,)
    """
    check1 = np.all(edge_function(a, b, p) >= 0, axis=-1)
    check2 = np.all(edge_function(b, c, p) >= 0, axis=-1)
    check3 = np.all(edge_function(c, a, p) >= 0, axis=-1)
    return check1 & check2 & check3


def triangulate_quad_mesh(F, S=None):
    """
    Triangulates a quad mesh.
    Each quad face (v0, v1, v2, v3) is split into two triangles:
      - Triangle 1: (v0, v1, v2)
      - Triangle 2: (v0, v2, v3)
    :param F: (m,4) np.ndarray of quad face indices.
    :param S: (m, x) some signal per face that needs change according to F
    :return: F_tri (m*2, 3) and S_tri (m*2, x) if S is not None
    """
    m = F.shape[0]
    F_tri = np.empty((m * 2, 3), dtype=F.dtype)
    F_tri[0::2] = F[:, [0, 1, 2]]
    F_tri[1::2] = F[:, [0, 2, 3]]
    if S is None:
        return F_tri
    else:
        S_tri = np.empty((m * 2, S.shape[-1]), dtype=S.dtype)
        S_tri[0::2] = S
        S_tri[1::2] = S
        return F_tri, S_tri


def duplicate_faces(f):
    """
    duplicates *every* face in the mesh, with flipped orientation (and appends it to the end of the tensor)
    note: will transfer to CPU if necessary with current implementation
    :param f: faces of the mesh (Nx3)
    :return: faces of the mesh with duplicated faces
    """
    if is_np(f):
        new_faces = f
    else:
        new_faces = f.detach().cpu().numpy()
    swapped_f = new_faces.copy()
    swapped_f[:, [1, 2]] = swapped_f[:, [2, 1]]
    f_new = np.concatenate([new_faces, swapped_f])
    if not is_np(f):
        f_new = torch.tensor(f_new, dtype=f.dtype, device=f.device)
    return f_new


def remove_duplicate_faces(f):
    """
    remove duplicate faces from a mesh
    note: will transfer to CPU if necessary with current implementation
    :param f: faces of the mesh (Nx3)
    :return: vertices and faces of the mesh without duplicate faces
    """
    if is_np(f):
        f_new = f
    else:
        f_new = f.detach().cpu().numpy()
    f_new = np.sort(f_new, axis=1)
    f_new = np.unique(f_new, axis=0)
    if not is_np(f):
        f_new = torch.tensor(f_new, dtype=f.dtype, device=f.device)
    return f_new


def get_aspect_ratio(v: torch.Tensor, f: torch.Tensor):
    """
    measure aspect ratio of all triangles using: (circumradius / 2inradius)
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
    return a * b * c / (8 * (s - a) * (s - b) * (s - c))


def get_face_areas(v, f, eps=1e-7):
    """
    :param v: vertex tensor Vx3
    :param f: face tensor Fx3
    :param eps: area thresold for degenerate face
    :return: tensor of face normals
    """
    face_vertices = v[f]
    v0 = face_vertices[:, 0, :]
    v1 = face_vertices[:, 1, :]
    v2 = face_vertices[:, 2, :]
    cross = torch.cross((v1 - v0), (v2 - v0), dim=1)
    norms = torch.norm(cross, dim=1, keepdim=True)
    face_areas = norms / 2
    face_areas[face_areas < eps] = eps
    return face_areas


def normalize_vertices(vertices, mode="unit_sphere"):
    """
    shift and resize mesh to fit into a bounding volume
    """
    eps = 1e-7
    if is_np(vertices):
        vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        if mode == "unit_sphere":
            vertices = vertices / (np.linalg.norm(vertices, axis=-1).max() + eps)
        elif mode == "unit_cube":
            vertices = vertices / (np.abs(vertices).max(dim=-1) + eps)
            raise NotImplementedError
    else:
        if mode == "unit_sphere":
            vertices -= (vertices.min(dim=0)[0] + vertices.max(dim=0)[0]) / 2
            vertices = vertices / (torch.norm(vertices, dim=-1).max() + eps)
        elif mode == "unit_cube":
            raise NotImplementedError
    return vertices


def calc_face_normals(
    vertices: torch.Tensor, faces: torch.Tensor, normalize: bool = False
):
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
        face_normals = torch.nn.functional.normalize(face_normals, eps=1e-6, dim=1)
    return face_normals  # F,3


def calc_vertex_normals(
    vertices: torch.Tensor, faces: torch.Tensor, face_normals: torch.Tensor = None
):
    """
    # V,3 first vertex may be unreferenced
    # F,3 long, first face may be all zero
    return # F,3, not normalized
    """
    F = faces.shape[0]
    if face_normals is None:
        face_normals = calc_face_normals(vertices, faces)
    vertex_normals = torch.zeros(
        (vertices.shape[0], 3, 3), dtype=vertices.dtype, device=vertices.device
    )  # V,C=3,3
    vertex_normals.scatter_add_(
        dim=0,
        index=faces[:, :, None].expand(F, 3, 3),
        src=face_normals[:, None, :].expand(F, 3, 3),
    )
    vertex_normals = vertex_normals.sum(dim=1)  # V,3
    return torch.nn.functional.normalize(vertex_normals, eps=1e-6, dim=1)


def get_face_centroids(v, f):
    return torch.mean(v[f], dim=-2)


def faces2edges_naive(faces: np.array):
    """
    a naive implementation in numpy to extract edges from a triangular / quad mesh.
    note: assumes manifold mesh with no borders
    :param faces: (n, 3) or (n, 4) np.uint32 of face indices
    :return edges (m, 2), f2e (n, 3 or 4) and e2f(m, 2)
    """
    edge_list = []
    faces_to_edges = []

    for face in faces:
        face_edges = []
        num_vertices = len(face)
        for i in range(num_vertices):
            v0, v1 = sorted((face[i], face[(i + 1) % num_vertices]))
            edge = (v0, v1)
            edge_list.append(edge)
            face_edges.append(edge)
        faces_to_edges.append(face_edges)

    # Convert to numpy array and find unique edges
    edge_array = np.array(edge_list, dtype=np.int32)
    unique_edges, edge_indices = np.unique(edge_array, axis=0, return_inverse=True)

    # Map faces to their corresponding unique edges
    faces_to_edges = np.array(edge_indices).reshape(faces.shape)

    # initialize the edges_to_faces array with a default value and fill it in:
    num_unique_edges = unique_edges.shape[0]
    edges_to_faces = -np.ones((num_unique_edges, 2), dtype=np.int32)
    for face_idx, face_edge_indices in enumerate(faces_to_edges):
        for edge_idx in face_edge_indices:
            if edges_to_faces[edge_idx, 0] == -1:
                edges_to_faces[edge_idx, 0] = face_idx
            else:
                edges_to_faces[edge_idx, 1] = face_idx
    return unique_edges, faces_to_edges, edges_to_faces


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
    sorted_edges, _ = full_edges.sort(dim=-1)  # F*3,2 todo min/max faster?

    # make unique edges
    edges, full_to_unique = torch.unique(
        input=sorted_edges, return_inverse=True, dim=0
    )  # (E,2),(F*3)
    E = edges.shape[0]
    face_to_edge = full_to_unique.reshape(F, 3)  # F,3
    if not with_edge_to_face:
        return edges, face_to_edge

    is_right = full_edges[:, 0] != sorted_edges[:, 0]  # F*3
    edge_to_face = torch.zeros(
        (E, 2, 2), dtype=torch.long, device=faces.device
    )  # E,LR=2,S=2
    scatter_src = torch.cartesian_prod(
        torch.arange(0, F, device=faces.device), torch.arange(0, 3, device=faces.device)
    )  # F*3,2
    edge_to_face.reshape(2 * E, 2).scatter_(
        dim=0,
        index=(2 * full_to_unique + is_right)[:, None].expand(F * 3, 2),
        src=scatter_src,
    )  # E,LR=2,S=2
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
    :param vertices (V,3)
    :param faces (F,3) long
    """
    V, D = vertices.shape
    vertices = torch.concat(
        (torch.full((1, D), fill_value=torch.nan, device=vertices.device), vertices),
        dim=0,
    )
    faces = torch.concat(
        (torch.zeros((1, 3), dtype=torch.long, device=faces.device), faces + 1), dim=0
    )
    return vertices, faces


def remove_dummies(vertices: torch.Tensor, faces: torch.Tensor):
    """
    remove dummy elements added with prepend_dummies()
    :param vertices (V,3) - first vertex all nan and unreferenced
    :param faces (F,3) long - first face all zeros
    :return: V-1,D and F-1,3
    """
    return vertices[1:], faces[1:] - 1


def ray_sphere_intersection(sphere_origin, sphere_radius, ray_origin, ray_direction):
    """
    returns the distance along a ray to the intersection point between a sphere and a ray
    # sphere_origin, sphere_radius, ray_origin, ray_direction: # B,3
    # returns: # B
    """
    radius = sphere_radius
    radius2 = radius**2
    center = sphere_origin
    L = center - ray_origin
    tca = torch.bmm(L[:, None, :], ray_direction[:, :, None]).squeeze()
    tca[tca < 0] = float("Inf")
    d2 = torch.bmm(L[:, None, :], L[:, :, None]).squeeze() - tca * tca
    d2[d2 > radius2] = -float("inf")
    thc = torch.sqrt(radius2 - d2)
    t0 = tca - thc
    t1 = tca + thc
    mask = t0 > t1
    t0[mask] = t1[mask]
    return t0


def qslerp(qa, qb, t):
    """
    interpolates between two quanternions
    :param qa: first quaternions
    :param qb: second quaternions
    :param t: number between 0 and 1 indicating the interpolation factor
    :return: b x 4 array of quaternions
    """
    qm = np.zeros_like(qa)
    cosHalfTheta = qa.dot(qb)
    if abs(cosHalfTheta) >= 1.0:  # theta = 0 degrees and we can return qa
        qm = qa
        return qm
    halfTheta = np.arccos(cosHalfTheta)
    sinHalfTheta = np.sqrt(1.0 - cosHalfTheta * cosHalfTheta)
    if (
        np.fabs(sinHalfTheta) < 0.001
    ):  # if theta = 180 degrees then result is not fully defined
        qm = qa * 0.5 + qb * 0.5
        return qm
    ratioA = np.sin((1 - t) * halfTheta) / sinHalfTheta
    ratioB = np.sin(t * halfTheta) / sinHalfTheta
    qm = qa * ratioA + qb * ratioB
    return qm


def ray_ray_intersection(oa, da, ob, db):
    """
    returns points closest to each corresponding ray of form o+t*d in the bundle
    :param oa: origin of ray a (b x 3)
    :param da: direction of ray a (b x 3)
    :param ob: origin of ray b (b x 3)
    :param db: direction of ray b (b x 3)
    :return: points closest to each ray (b x 3) and a weight factor that goes to 0 if the lines are parallel
    """
    batched_input = True
    if oa.ndim == 1:
        oa = oa[None, :]
        batched_input = False
    if da.ndim == 1:
        da = da[None, :]
        batched_input = False
    if ob.ndim == 1:
        ob = ob[None, :]
        batched_input = False
    if db.ndim == 1:
        db = db[None, :]
        batched_input = False
    oa, ob, da, db = broadcast_batch(oa, ob, da, db)
    da = da / np.linalg.norm(da, axis=1, keepdims=True)
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    c = np.cross(da, db)
    denom = np.linalg.norm(c, axis=-1) ** 2
    t = ob - oa
    stackedb = np.concatenate([t[:, None, :], db[:, None, :], c[:, None, :]], axis=1)
    stackeda = np.concatenate([t[:, None, :], da[:, None, :], c[:, None, :]], axis=1)
    ta = np.linalg.det(stackedb) / (denom + 1e-10)
    tb = np.linalg.det(stackeda) / (denom + 1e-10)
    ta[ta < 0] = 0
    tb[tb < 0] = 0
    points = (oa + ta[:, None] * da + ob + tb[:, None] * db) * 0.5
    if not batched_input:
        points = points[0]
    return points, denom


def get_center_of_attention(c2w):
    """
    find a central point of a batch of c2w transforms 4x4 they are all looking at.
    note: point is weighted by distance between lines.
    :param camera_poses: n x 4 x 4 np array of c2w matrices
    :return: the center of attention in 3d world coordinates
    """
    if c2w.ndim != 3:
        raise ValueError("c2w must be a 3d array of 4x4 matrices")
    if c2w.shape[1] != 4 or c2w.shape[2] != 4:
        raise ValueError("c2w must be a 3d array of 4x4 matrices")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    ps = []
    rays = []
    for f in c2w:
        mf = f[0:3, :]
        for g in c2w:
            mg = g[0:3, :]
            p, w = ray_ray_intersection(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
                # ps.append(p)
                # rays.append([mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2]])
    totp /= totw
    return totp  # np.array(rays), np.array(ps)


def scale_poses(c2w, n=1.0):
    """
    scales c2w transforms so that the average distance to the center of attention is n
    :param c2w: n x 4 x 4 np array of c2w matrices
    :param n: the average distance to the center of attention
    :return: the scaled c2w matrices
    """
    avglen = np.mean(np.linalg.norm(c2w[:, 0:3, 3], axis=-1))
    # print("avg camera distance from origin", avglen)
    c2w[:, 0:3, 3] *= n / avglen
    return c2w, avglen


def merge_meshes(v1, f1, v2=None, f2=None):
    """merge two meshes into one"""
    if v2 is None or f2 is None:
        return v1, f1
    v = np.concatenate([v1, v2], axis=0)
    f = np.concatenate([f1, f2 + v1.shape[0]], axis=0)
    return v, f


def find_princple_componenets(v: torch.Tensor):
    """
    finds the principle components of a points nx3
    :param v: nx3 torch tensor of points
    :return: the principle components (column major)
    """
    if type(v) == torch.Tensor:
        cov = v.T @ v
        _, vecs = torch.linalg.eig(cov)
        if torch.imag(vecs).any():
            raise ValueError("imaginary eigenvectors")
        vecs = torch.real(vecs)
        if torch.det(vecs) < 0:
            swap_axis = torch.tensor(
                [[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=v.dtype, device=v.device
            )
            vecs = swap_axis @ vecs
    else:
        raise TypeError("v must be a torch tenor")
    return vecs


def remove_unreferenced_vertices(v, f):
    """
    removes unreferenced vertices from a mesh
    :param v: Vx3 np array of vertices
    :param f: Fx3 np array of faces
    :return: the new vertices and faces, and a map from new to old indices (-1 if unreferenced)
    """
    referenced = np.zeros((v.shape[0]), dtype=bool)
    referenced[f.flatten()] = True
    idx = -1 * np.ones((v.shape[0]), dtype=np.int64)
    idx[referenced] = np.arange(np.sum(referenced), dtype=np.int64)
    f = idx[f.flatten()].reshape((-1, f.shape[1]))
    v = v[referenced]
    return v, f, idx


def get_edges(f: np.ndarray):
    """
    given a numpy array of faces of a triangular mesh F x 3, returns a numpy array of edges
    :param f: F x 3 numpy array of faces
    :return: E x 2 numpy array of edges
    """
    e1 = np.concatenate((f[:, 0:1], f[:, 1:2]), axis=1)
    e2 = np.concatenate((f[:, 1:2], f[:, 2:3]), axis=1)
    e3 = np.concatenate((f[:, 2:3], f[:, 0:1]), axis=1)
    e = np.concatenate((e1, e2, e3), axis=0)
    e = np.sort(e, axis=-1)
    e = np.unique(e, axis=0)
    return e


def edge_contraction(v, f, edge_to_contract, new_v_location):
    """
    contracts an edge in a mesh and moves the vertex to a new location
    :param v: V x 3 numpy array of vertices
    :param f: F x 3 numpy array of faces
    :param edge_to_contract: 2, numpy array of the edge to contract
    :param new_v_location: 3, numpy array of the new vertex location
    :return: the contracted mesh
    """
    v1 = edge_to_contract[0]
    v2 = edge_to_contract[1]
    # replace all instances of v1 with v2 in f
    f[f == v1] = v2
    # remove degenerate triangles
    degenerate_mask = (np.diff(np.sort(f, axis=-1)) == 0).any(axis=-1)
    f = f[~degenerate_mask]
    # place v2 in v_hat
    v[v2] = new_v_location
    # remove v1 from v
    v, f, _ = remove_unreferenced_vertices(v, f)
    return v, f


def clean_infinite_vertices(v, f):
    """
    removes vertices that are infinite / nan, and all their incident faces
    :param v: V x 3 numpy array of vertices
    :param f: F x 3 numpy array of faces
    :return: the cleaned mesh
    """
    finite_mask = np.isfinite(v).any(axis=-1)  # finite mask
    if ~np.all(finite_mask):
        f = f[finite_mask[f].all(axis=-1)]  # remove faces with infinite vertices
        v, f, _ = remove_unreferenced_vertices(v, f)  # update vertex indices
    return v, f
