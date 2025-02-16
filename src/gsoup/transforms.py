import numpy as np
from .core import (
    to_44,
    compose_rt,
    normalize,
    broadcast_batch,
    is_np,
    map_range,
    to_torch,
)
import torch


def ndc_to_screen(ndc, width, height):
    """
    convert 2D NDC (normalized device coordiantes) to screen space
    :param ndc: the b x 2 coordinates
    :param width: the width of the screen in pixels
    :param height: the height of the screen in pixels
    """
    size = np.array([width, height], dtype=np.float32)[None, :]
    ndc_norm = map_range(ndc, -1.0, 1.0, 0.0, 1.0)
    if is_np(ndc):
        screen = (ndc_norm * size).astype(np.int32)
    else:
        screen = (
            ndc_norm * to_torch(size, device=ndc_norm.device, dtype=torch.float32)
        ).type(torch.uint32)
    return screen


def screen_to_ndc(screen, width, height):
    """
    convert screen space coordinates to NDC
    :param ndc: the b x 2 coordinates
    :param width: the width of the screen in pixels
    :param height: the height of the screen in pixels
    """
    size = np.array([width - 1, height - 1], dtype=np.float32)[None, :]
    if not is_np(screen):
        size = to_torch(size, device=screen.device)
    ndc = map_range(screen, 0.0, size, -1.0, 1.0)
    return ndc


def sincos(a, degrees=True):
    """
    sin and cos of an angle
    :param a: angle
    :param degrees: if True, a is in degrees
    """
    if degrees:
        a = np.deg2rad(a)
    return np.sin(a), np.cos(a)


def translate(t):
    """
    creates a translation matrix from a translation vector
    :param t: translation vector or batch of translation vectors
    :return: 4x4 translation matrix
    """
    if t.shape[-1] != 3:
        raise ValueError("translation vector must be 3d")
    if t.ndim == 1:
        t = t[None, :]
    mat = np.concatenate((np.eye(3)[None, :, :], t[:, :, None]), axis=-1)
    return to_44(mat)


def scale(s):
    """
    creates a scaling matrix from a scaling vector
    :param s: scaling vector or batch of scaling vectors
    :return: nx4x4 scaling matrix
    """
    if s.shape[-1] != 3:
        raise ValueError("translation vector must be 3d")
    if s.ndim == 1:
        s = s[None, :]
    mat = np.diag(s)
    return to_44(mat)


def rotate(a, r):
    """
    creates a rotation matrix from an angle a and an axis of rotation r
    """
    s, c = sincos(a)
    r = normalize(r)
    nc = 1 - c
    x, y, z = r
    return np.array(
        [
            [x * x * nc + c, x * y * nc - z * s, x * z * nc + y * s, 0],
            [y * x * nc + z * s, y * y * nc + c, y * z * nc - x * s, 0],
            [x * z * nc - y * s, y * z * nc + x * s, z * z * nc + c, 0],
            [0, 0, 0, 1],
        ]
    )


def rotx(a, degrees=True):
    """
    creates a homogeneous 3D rotation matrix around the x axis
    a: angle
    degrees: if True, a is in degrees, else radians
    """
    s, c = sincos(a, degrees)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def roty(a, degrees=True):
    """
    creates a homogeneous 3D rotation matrix around the y axis
    a: angle
    degrees: if True, a is in degrees, else radians
    """
    s, c = sincos(a, degrees)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def rotz(a, degrees=True):
    """
    creates a homogeneous 3D rotation matrix around the z axis
    a: angle
    degrees: if True, a is in degrees, else radians
    """
    s, c = sincos(a, degrees)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def find_rigid_transform(A, B):
    """
    finds best 3d rigid transformation between pc a and pc b (in the least squares sense) based on "Least-Squares Rigid Motion Using SVD"
    :param A: point cloud a
    :param B: point cloud b
    :return: W such that W@A = B
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_mean = A - centroid_A
    B_mean = B - centroid_B

    H = A_mean.T @ B_mean
    U, S, Vt = np.linalg.svd(H)

    flip = np.linalg.det(Vt.T @ U.T)
    ones = np.identity(len(Vt))
    ones[-1, -1] = flip
    R = Vt.T @ ones @ U.T
    t = centroid_B - R @ centroid_A
    return compose_rt(R[None, :], t[None, :], square=True)


def find_affine_transformation(A, B):
    """
    finds best 3d affine transformation between pc a and pc b (in the least squares sense)
    :param A: point cloud a
    :param B: point cloud b
    :return: W such that W@A = B
    """
    # assemble A matrix for least squares
    assert len(A) >= 4
    new_A = np.c_[A, np.ones(len(A))]
    new_A = np.repeat(np.tile(new_A, (1, 3)), 3, axis=0)
    # now A is 3nx12
    # zero columns depending on index of point
    new_A[::3, 4:] = 0
    new_A[1::3, :4] = 0
    new_A[1::3, 8:] = 0
    new_A[2::3, :8] = 0
    new_B = B.flatten()
    if len(A) == 4:
        # solve linear system
        W = np.linalg.inv(new_A) @ new_B
    else:
        W = np.linalg.lstsq(new_A, new_B, rcond=None)[0]
    return to_44(W.reshape(3, 4))


def decompose_affine(A44):
    """
    Decompose 4x4 homogenous affine matrix into parts.
    The parts are translations, rotations, zooms, shears.
    Returns
    -------
    T : array, shape (3,)
       Translation vector
    R : array shape (3,3)
        rotation matrix
    Z : array, shape (3,)
       scale vector.  May have one negative zoom to prevent need for negative
       determinant R matrix above
    S : array, shape (3,)
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix (type ``striu``).
    Implementation of "Decomposing a matrix into simple transformations".
    """
    A44 = np.asarray(A44)
    T = A44[:-1, -1]
    RZS = A44[:-1, :-1]
    # compute scales and shears
    M0, M1, M2 = np.array(RZS).T
    # extract x scale and normalize
    sx = np.sqrt(np.sum(M0**2))
    M0 /= sx
    # orthogonalize M1 with respect to M0
    sx_sxy = np.dot(M0, M1)
    M1 -= sx_sxy * M0
    # extract y scale and normalize
    sy = np.sqrt(np.sum(M1**2))
    M1 /= sy
    sxy = sx_sxy / sx
    # orthogonalize M2 with respect to M0 and M1
    sx_sxz = np.dot(M0, M2)
    sy_syz = np.dot(M1, M2)
    M2 -= sx_sxz * M0 + sy_syz * M1
    # extract z scale and normalize
    sz = np.sqrt(np.sum(M2**2))
    M2 /= sz
    sxz = sx_sxz / sx
    syz = sy_syz / sy
    # Reconstruct rotation matrix, ensure positive determinant
    Rmat = np.array([M0, M1, M2]).T
    if np.linalg.det(Rmat) < 0:
        sx *= -1
        Rmat[:, 0] *= -1
    return T, Rmat, np.array([sx, sy, sz]), np.array([sxy, sxz, syz])


def invert_rigid(rt, square=False):
    """
    inverts a rigid transformation (n, 4, 4) efficiently
    rt: (n, 4, 4) batch of rigid transformations
    square: if false, will return a (n, 3, 4) by removing the last row
    """
    if rt.ndim != 3:
        return ValueError("rt must have 3 dimensions")
    if rt.shape[1] != 4 or rt.shape[2] != 4:
        return ValueError("rt must have shape (n, 4, 4)")
    if is_np(rt):
        R = np.transpose(rt[:, :3, :3], axes=(0, 2, 1))
        t = (-R @ rt[:, :-1, -1].T)[:, :, 0]
        return compose_rt(R, t, square)
    else:
        return NotImplementedError


def look_at_np(eye, at, up, opengl=False):
    """
    returns a batch of look_at transforms 4x4 (camera->world, the inverse of a ModelView matrix)
    will broadcast upon batch dimension if necessary.
    :param eye: n x 3 from vectors in world space
    :param at: n x 3 at vector in world space
    :param up: n x 3 up vector in world space
    :param opengl: if True, output will be in OpenGL coordinates (z backward, y up) otherwise (z forward, y down)
    :return: n x 4 x 4 transformation matrices (camera->world, the inverse of a ModelView matrix)
    """
    eye, at, up = broadcast_batch(eye, at, up)
    forward = at - eye
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)
    rot = np.concatenate((right[..., None], up[..., None], forward[..., None]), axis=-1)
    c2w = compose_rt(rot, eye, square=True)
    if opengl:
        c2w = opencv_c2w_to_opengl_c2w(c2w)
    return c2w


def look_at_torch(
    eye: torch.Tensor,  # 3
    at: torch.Tensor,  # 3
    up: torch.Tensor,  # 3
    opengl: bool = False,
) -> torch.Tensor:  # 4,4
    """
    creates a lookat transform matrix
    :param eye: where the camera is
    :param at: where the camera is looking
    :param up: the up vector of world space
    :param opengl: if True, output will be in OpenGL coordinates (z backward, y up) otherwise (z forward, y down)
    :return: 4x4 lookat transform matrix
    """
    # todo batch support
    z = (at - eye).type(torch.float32)
    z /= torch.norm(z)
    x = torch.linalg.cross(z, up).type(torch.float32)
    x /= torch.norm(x)
    y = torch.linalg.cross(z, x).type(torch.float32)
    y /= torch.norm(y)
    c2w = torch.eye(4, device=z.device)
    c2w[:3, :3] = torch.stack([x, y, z], dim=1)
    c2w[:3, 3] = eye
    if opengl:
        c2w = opencv_c2w_to_opengl_c2w(c2w)
    return c2w


def orthographic_projection(l, r, b, t, n, f):
    """
    creates an orthographic projection matrix (OpenGL convention)
    :param l: left
    :param r: right
    :param b: bottom
    :param t: top
    :param n: near
    :param f: far
    :return: 4x4 orthographic projection matrix
    """
    dx = r - l
    dy = t - b
    dz = f - n
    rx = -(r + l) / (r - l)
    ry = -(t + b) / (t - b)
    rz = -(f + n) / (f - n)
    return np.array(
        [
            [2.0 / dx, 0, 0, rx],
            [0, 2.0 / dy, 0, ry],
            [0, 0, -2.0 / dz, rz],
            [0, 0, 0, 1],
        ]
    )


def perspective_projection(fovy=45, aspect=1.0, n=0.1, f=100.0):
    """
    creates a perspective projection matrix (OpenGL convention)
    :param fovy: field of view in y direction
    :param aspect: aspect ratio
    :param n: near plane
    :param f: far plane
    :return: 4x4 projection matrix
    """
    s = 1.0 / np.tan(np.deg2rad(fovy) / 2.0)
    sx, sy = s / aspect, s
    zz = (f + n) / (n - f)
    zw = 2 * f * n / (n - f)
    return np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, zz, zw], [0, 0, -1, 0]])


def frustum_projection(x0, x1, y0, y1, z0, z1):
    """
    creates a projection matrix from a frustum (OpenGL convention)
    :param x0: left
    :param x1: right
    :param y0: bottom
    :param y1: top
    :param z0: near
    :param z1: far
    :return: 4x4 projection matrix
    """
    a = (x1 + x0) / (x1 - x0)
    b = (y1 + y0) / (y1 - y0)
    c = -(z1 + z0) / (z1 - z0)
    d = -2 * z1 * z0 / (z1 - z0)
    sx = 2 * z0 / (x1 - x0)
    sy = 2 * z0 / (y1 - y0)
    return np.array([[sx, 0, a, 0], [0, sy, b, 0], [0, 0, c, d], [0, 0, -1, 0]])


def opengl_project_from_opencv_intrinsics(
    opencv_intrinsics, width, height, near=0.1, far=100.0, window_coords_inverted=False
):
    """
    given a matrix K from opencv, returns the corresponding projection matrix for OpenGL ("Eye/Camera/View space -> Clip Space")
    :param opencv_intrinsics: 3x3 intrinsics matrix from opencv
    :param width: width of the image
    :param height: height of the image
    :param near: near plane
    :param far: far plane
    :param window_coords_inverted: if True, the y axis is inverted (y down)
    :return: 4x4 projection matrix for OpenGL (note: column major)
    """
    fx = opencv_intrinsics[0, 0]
    fy = opencv_intrinsics[1, 1]
    cx = opencv_intrinsics[0, 2]
    cy = opencv_intrinsics[1, 2]
    factor = 1.0
    if window_coords_inverted:
        factor = -1.0
    opengl_mtx = np.array(
        [
            [2 * fx / width, 0.0, (width - 2 * cx) / width, 0.0],
            [0.0, factor * 2 * fy / height, factor * (height - 2 * cy) / height, 0.0],
            [0.0, 0.0, (-far - near) / (far - near), -2.0 * far * near / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    return opengl_mtx


def opencv_intrinsics_from_opengl_project(opengl_project, width, height):
    """
    given a projection matrix from OpenGL ("Eye/Camera/View space -> Clip Space"), returns a matrix K for opencv
    :param opengl_project: 4x4 projection matrix from OpenGL
    :param width: width of the image
    :param height: height of the image
    :return: 3x3 intrinsics matrix for opencv
    note: assumes camera center is middle of image
    """
    fx = opengl_project[0, 0] * width / 2
    fy = opengl_project[1, 1] * height / 2
    cx = width / 2
    cy = height / 2
    opencv_mtx = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1]])
    return opencv_mtx


def opengl_c2w_to_opencv_c2w(opengl_transforms, is_column_major=False):
    """
    given a modelview matrix (World space->Eye/Camera/View space) where z is backward and y is up,
    converts its coordinate system to opencv convention (z forward, y down)
    :param opengl_transforms: 4x4 modelview matrix or batch of 4x4 modelview matrices
    :param is_column_major: whether the input matrix is column major or row major
    :return: 4x4 modelview matrix in opencv convention
    """
    if opengl_transforms.ndim == 2:
        my_transforms = opengl_transforms[None, ...]
    else:
        my_transforms = opengl_transforms
    if my_transforms.shape[1:] != (4, 4):
        raise ValueError("transform must be 4x4 or batch of 4x4")
    if is_column_major:
        if is_np(my_transforms):
            my_transforms = my_transforms.transpose(0, 2, 1)
        else:
            my_transforms = my_transforms.permute(0, 2, 1)
        my_transforms[:, 1, :] *= -1
        my_transforms[:, 2, :] *= -1
    else:
        my_transforms[:, :, 1] *= -1
        my_transforms[:, :, 2] *= -1
    return my_transforms.reshape(opengl_transforms.shape)


def opencv_c2w_to_opengl_c2w(opencv_transform, to_column_major=False):
    """
    converts coordinates of "vision" (opencv) to OpenGL coordinates by flipping y and z axes
    """
    return opengl_c2w_to_opencv_c2w(opencv_transform, to_column_major)


def create_random_cameras_on_unit_sphere(
    n_cams, radius, normal=None, opengl=False, device="cpu"
):
    """
    creates a batch of world2view ("ModelView" matrix) and view2clip ("Projection" matrix) transforms on a unit sphere looking at the center
    :param n_cams: number of cameras
    :param radius: radius of the sphere
    :param normal: if provided, only the hemisphere in the direction of the normal is sampled
    :param opengl: if True, the coordinate system is converted to openGL convention (z backward, y up)
    :param device: device to put the tensors on
    :return: world2view (world2cam), view2clip (requires further processing if opengl=False)
    """
    locs = torch.randn((n_cams, 3), device=device)
    locs = torch.nn.functional.normalize(locs, dim=1, eps=1e-6)
    if normal is not None:
        if normal.ndim == 1:
            normal = normal[None, :]
        normal = torch.nn.functional.normalize(normal, dim=-1, eps=1e-6)
        dot_product = (locs[:, None, :] @ normal[:, :, None]).squeeze()
        locs[dot_product < 0] *= -1
    locs = locs * radius
    matrices = torch.empty((n_cams, 4, 4), dtype=torch.float32, device=device)
    for i in range(len(locs)):
        matrices[i] = look_at_torch(
            locs[i],
            torch.zeros(3, dtype=torch.float32, device=device),
            torch.tensor([0.0, 0.0, 1.0], device=device),
            opengl=opengl,
        )
    v2w = matrices  # c2w
    w2v = torch.inverse(v2w)
    v2c = perspective_projection()
    v2c = np.tile(v2c[None, :], (n_cams, 1, 1))
    v2c = torch.tensor(v2c, dtype=torch.float32, device=device)
    return w2v, v2c


def pixels_in_world_space(camera_wh, K, Rt):
    """
    computes pinhole camera pixel locations in world space
    :param camera_wh: resolution (width, height) as a tuple
    :param K: the camera instrinsics matrix 3x3
    :param Rt: the camera to world matrix 4x4
    :return: np array n x 3 for pixels locations (in world space)
    """
    x, y = np.meshgrid(
        np.arange(camera_wh[0], dtype=np.float32),
        np.arange(camera_wh[1], dtype=np.float32),
        indexing="xy",
    )
    x = x.reshape(-1)[:, None]
    y = y.reshape(-1)[:, None]
    ones = np.ones(len(x))[:, None]
    pixels_image = np.concatenate((x, y, ones), axis=-1)
    pixels_camera = np.linalg.inv(K) @ pixels_image.T
    pixels_camera_hom = np.concatenate(
        (pixels_camera.T, np.ones(len(pixels_camera.T))[:, None]), axis=-1
    )
    pixels_world = Rt @ pixels_camera_hom.T
    pc = pixels_world.T[:, :3]
    return pc


def random_qvec(n: int):
    """
    generate random quaternions representing rotations
    :param n: Number of quaternions in a batch to return.
    :return: Quaternions as tensor of shape (N, 4).
    """
    o = np.random.randn(n, 4)
    s = (o * o).sum(1)
    denom = np.copysign(np.sqrt(s), o[:, 0])[:, None]
    o = o / denom
    return o


def random_affine(ang_range=20.0, trans_range=10.0, scale=2.0):
    """
    generates a random 2D affine transformation matrix
    """
    R = np.eye(3)
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    tx = trans_range * np.random.uniform() - trans_range / 2
    ty = trans_range * np.random.uniform() - trans_range / 2
    sx = np.random.rand() * scale
    sy = np.random.rand() * scale
    R[0, 0] = np.cos(ang_rot * np.pi / 180)
    R[0, 1] = -np.sin(ang_rot * np.pi / 180)
    R[1, 0] = np.sin(ang_rot * np.pi / 180)
    R[1, 1] = np.cos(ang_rot * np.pi / 180)
    T = np.eye(3)
    T[0, 2] = tx
    T[1, 2] = ty
    S = np.eye(3)
    S[0, 0] = sx
    S[1, 1] = sy
    H = T @ S @ R
    return H


def random_perspective():
    """
    generates a random 2D perspective transformation matrix
    """
    P = random_affine()
    P[2, 0] = np.random.rand() / 100
    P[2, 1] = np.random.rand() / 100
    return P


def random_vectors_on_sphere(n, normal=None, device="cpu"):
    """
    create a batch of uniformly distributed random unit vectors on a sphere
    note: if normal is provided, returns random unit vectors on the hemisphere around the normal, but isn't uniform anymore.
    :param n: number of vectors
    :param normal: normals to orient the hemisphere (,3) or (n,3)
    :param device: device to put the tensors on
    :return: tensor of shape (n, 3)
    """
    locs = torch.randn((n, 3), device=device)
    locs = torch.nn.functional.normalize(locs, dim=1, eps=1e-6)
    if normal is not None:
        if normal.ndim == 1:
            normal = normal[None, :]
        normal = torch.nn.functional.normalize(normal, dim=-1, eps=1e-6)
        dot_product = (locs[:, None, :] @ normal[:, :, None]).squeeze()
        locs[dot_product < 0] *= -1
    return locs


def vec2skew(v):
    """
    returns the skew operator matrix given a vector
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    return batch_vec2skew(v[None, :])[0]


def batch_vec2skew(v):
    """
    returns a batch of skew operator matrices given a batch of vectors
    :param v:  (B, 3) torch tensor
    :return:   (B, 3, 3)
    """
    zero = torch.zeros_like(v[:, 0:1])
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], dim=1)  # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], dim=1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], dim=1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (B, 3, 3)
    return skew_v  # (B, 3, 3)


def batch_rotvec2mat(r: torch.Tensor):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    if r.ndim != 2:
        raise ValueError("r must be 2D tensor.")
    skew_r = batch_vec2skew(r)  # (3, 3)
    norm_r = r.norm(dim=-1, keepdim=True)[:, :, None] + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)[None, :]
    R = (
        eye
        + (torch.sin(norm_r) / norm_r) * skew_r
        + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    )
    return R


def rotvec2mat(r: torch.Tensor):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    return batch_rotvec2mat(r[None, :])[0]


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


def qvec2mat(qvecs):
    """
    Converts a batch of quaternions to a batch of rotation matrices
    :param qvec: tensor of shape (n, 4) where real part is first [a + bi + cj + dk] => [a, b, c, d]
    :return: rotation matrix (n, 3, 3)
    """
    if qvecs.ndim != 2:
        raise ValueError("qvec should have ndim == 2")
    if qvecs.shape[-1] != 4:
        raise ValueError("quaternions must be of shape (..., 4)")
    if is_np(qvecs):
        r, i, j, k = [x[0] for x in np.split(qvecs, 4, -1)]
        stacking_func = np.stack
    else:
        r, i, j, k = torch.unbind(qvecs, -1)
        stacking_func = torch.stack
    two_s = 2.0 / (qvecs * qvecs).sum(-1)
    o = stacking_func(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        -1,
    )
    return o.reshape(qvecs.shape[:-1] + (3, 3))


def mat2qvec_numpy(R: np.array):
    """
    converts a rotation matrix to a quaternion
    :param R: np array of size 3x3
    :return: qvec (4,) xyzw
    """
    q = np.empty((R.shape[0], 4), dtype=R.dtype)
    trace = np.trace(R)
    expanded_trace = np.concatenate((np.diag(R), trace), axis=0)
    choice = np.argmax(expanded_trace, axis=-1)
    mask = choice != 3
    i = choice[mask]
    j = (i + 1) % 3
    k = (j + 1) % 3
    ii = np.concatenate((i, i), axis=1)
    ij = np.concatenate((i, j), axis=1)
    ik = np.concatenate((i, k), axis=1)
    jk = np.concatenate((j, k), axis=1)
    q[:, 0] = R[:, 2, 1] - R[:, 1, 2]
    q[:, 1] = R[:, 0, 2] - R[:, 2, 0]
    q[:, 2] = R[:, 1, 0] - R[:, 0, 1]
    q[:, 3] = 1 + trace
    q[mask, 0] = 1 - trace + 2 * R[ii]
    q[mask, 1] = R[np.flip(ij)] + R[ij]
    q[mask, 2] = R[np.flip(ik)] + R[ik]
    q[mask, 3] = R[np.flip(jk)] - R[jk]
    raise NotImplementedError("please verify this function implementation before usage")
    # Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    # K = np.array([
    #     [Rxx - Ryy - Rzz, 0, 0, 0],
    #     [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
    #     [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
    #     [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    # eigvals, eigvecs = np.linalg.eigh(K)
    # qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    # if qvec[0] < 0:
    #     qvec *= -1
    # return qvec
    return q


def mat2qvec(matrix):
    """
    Converts a batch of rotation matrices to a batch of quaternions
    :param matrix: tensor of size nx3x3
    :return: qvec: tensor of size nx4 where real part is first (a + bi + cj + dk) => (a, b, c, d)
    implementation is exact copy of pytorch3d implementation
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    )
    q_abs = torch.zeros_like(q)
    positive_mask = q > 0
    q_abs[positive_mask] = torch.sqrt(q[positive_mask])
    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
