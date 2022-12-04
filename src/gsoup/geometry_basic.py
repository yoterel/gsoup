import torch
import numpy as np

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
    return a*b*c/(8*(s-a)*(s-b)*(s-c))

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
    if type(vertices) == np.ndarray:
        vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        if mode == "unit_sphere":
            vertices = vertices / (np.linalg.norm(vertices, axis=-1).max() + eps)
        elif mode == "unit_cube":
            vertices = vertices / (np.abs(vertices).max(dim=-1) + eps)
            raise NotImplementedError
        
    elif type(vertices) == torch.Tensor:
        if mode == "unit_sphere":
            vertices -= (vertices.min(dim=0)[0] + vertices.max(dim=0)[0]) / 2
            vertices = vertices / (torch.norm(vertices, dim=-1).max() + eps)
        elif mode == "unit_cube":
            raise NotImplementedError
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

def get_face_centroids(v, f):
    return torch.mean(v[f], dim=-2)

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

def ray_sphere_intersection(sphere_origin, sphere_radius, ray_origin, ray_direction):
    """
    returns the distance along a ray to the intersection point between a sphere and a ray
    # sphere_origin, sphere_radius, ray_origin, ray_direction: # B,3
    # returns: # B
    """
    radius = sphere_radius
    radius2 = radius ** 2
    center = sphere_origin
    L = (center - ray_origin)
    tca = torch.bmm(L[:, None, :], ray_direction[:, :, None]).squeeze()
    tca[tca < 0] = float("Inf")
    d2 = torch.bmm(L[:, None, :], L[:, :, None]).squeeze() - tca * tca
    d2[d2 > radius2] = -float('inf')
    thc = torch.sqrt(radius2 - d2)
    t0 = tca - thc
    t1 = tca + thc
    mask = t0 > t1
    t0[mask] = t1[mask]
    return t0

def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([zero, -v[2:3], v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([v[2:3], zero, -v[0:1]])
    skew_v2 = torch.cat([-v[1:2], v[0:1], zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def rotvec2mat(r: torch.Tensor):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r ** 2) * (skew_r @ skew_r)
    return R


def mat2rotvec(r: torch.Tensor):
    """SO(3) matrix to so(3) vector
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    e, v = torch.linalg.eig(r)
    rotvec = v.real[:, torch.isclose(e.real, torch.ones(1))].squeeze()
    up = torch.tensor([0, 1, 0], dtype=torch.float)
    # ortho = torch.cross(rotvec, up)
    # ortho = ortho / ortho.norm(keepdim=True)
    # angle1 = torch.arccos(ortho.dot(r @ ortho))
    angle2 = torch.arccos((torch.trace(r) - 1) / 2)
    raise NotImplementedError("please verify this function implementation before usage")
    return rotvec * angle2


def qvec2rotmat(qvec: np.array):
    """
    converts a Quaternions to a rotation matrix
    :param qvec: np array (4,)
    :return: 3x3 np array
    """
    rot_mat = np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])
    return rot_mat


def rotmat2qvec(R: np.array):
    """
    converts a rotation matrix to a Quaternions
    :param R: np array of size 3x3
    :return: qvec (4,)
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qslerp(qa, qb, t):
    """
    interpolates between two quanternions
    :param qa: 
    :param qb:
    :param t:
    :return:
    """
    qm = np.zeros_like(qa)
    cosHalfTheta = qa.dot(qb)
    if abs(cosHalfTheta) >= 1.0:  # theta = 0 degrees and we can return qa
        qm = qa
        return qm
    halfTheta = np.arccos(cosHalfTheta)
    sinHalfTheta = np.sqrt(1.0 - cosHalfTheta * cosHalfTheta)
    if np.fabs(sinHalfTheta) < 0.001:  # if theta = 180 degrees then result is not fully defined
        qm = qa*0.5 + qb*0.5
        return qm
    ratioA = np.sin((1 - t) * halfTheta) / sinHalfTheta
    ratioB = np.sin(t * halfTheta) / sinHalfTheta
    qm = qa*ratioA + qb*ratioB
    return qm


def ray_ray_intersection(oa, da, ob, db):
    """
    returns point closest to both rays of form o+t*d,
    and a weight factor that goes to 0 if the lines are parallel
    :param oa:
    :param da:
    :param ob:
    :param db:
    :return:
    """
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta < 0:
        ta = 0
    if tb < 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def get_center_of_attention(c2w):
    """
    find a central point of a batch of c2w transforms 4x4 they are all looking at.
    note: point is weighted by distance between lines.
    :param camera_poses: n x 4 x 4 np array of c2w matrices
    :return: the center of attention in 3d world coordinates
    """
    if c2w.ndim != 3:
        raise ValueError('c2w must be a 3d array of 4x4 matrices')
    if c2w.shape[1] != 4 or c2w.shape[2] != 4:
        raise ValueError('c2w must be a 3d array of 4x4 matrices')
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
    return totp # np.array(rays), np.array(ps)

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
            raise ValueError('imaginary eigenvectors')
        vecs = torch.real(vecs)
        if torch.det(vecs) < 0:
            swap_axis = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=v.dtype, device=v.device)
            vecs = swap_axis @ vecs
    else:
        raise TypeError('v must be a torch tenor')
    return vecs

def remove_unreferenced_vertices(v, f):
    """
    removes unreferenced vertices from a mesh
    :param v: Vx3 torch tensor of vertices
    :param f: Fx3 torch tensor of faces
    :return: the new vertices and faces
    """
    referenced = np.zeros((v.shape[0]), dtype=np.bool)
    referenced[f.flatten()] = True
    idx = -1 * np.ones((v.shape[0]), dtype=np.int32)
    idx[referenced] = np.arange(np.sum(referenced), dtype=np.int32)
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
    f[f==v1] = v2
    # remove degenerate triangles
    degenerate_mask = (np.diff(np.sort(f, axis=-1)) == 0).any(axis=-1)
    f = f[~degenerate_mask] 
    # place v2 in v_hat
    v[v2] = new_v_location
    # remove v1 from v
    v, f, _ = remove_unreferenced_vertices(v, f)
    return v, f