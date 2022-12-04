import torch
import numpy as np
import torch.nn as nn

def icosehedron(scale=1.0, centered=True):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],

        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],

        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ]).astype(np.float32)
    if centered:
        vertices -= np.array([0.0, 0.0, 0.0])
    vertices *= scale
    faces = np.array([
        [ 0, 11,  5],
        [ 0,  5,  1],
        [ 0,  1,  7],
        [ 0,  7, 10],
        [ 0, 10, 11],

        [ 1,  5,  9],
        [ 5, 11,  4],
        [11, 10,  2],
        [10,  7,  6],
        [ 7,  1,  8],

        [ 3,  9,  4],
        [ 3,  4,  2],
        [ 3,  2,  6],
        [ 3,  6,  8],
        [ 3,  8,  9],

        [ 4,  9,  5],
        [ 2,  4, 11],
        [ 6,  2, 10],
        [ 8,  6,  7],
        [ 9,  8,  1],
    ]).astype(np.int32)
    return vertices, faces

def cube(scale=1.0, centered=True):
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
    vertices *= scale
    faces = np.array([
        [0, 2, 1],
        [0, 3, 2],
        [0, 1, 3],
        [1, 6, 3],
        [1, 4, 6],
        [1, 2, 4],
        [2, 5, 4],
        [2, 3, 5],
        [3, 6, 5],
        [5, 6, 7],
        [4, 5, 7],
        [4, 7, 6],
    ]).astype(np.int32)
    return vertices, faces

def get_gizmo_coords(scale=1.0):
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

def length(x):
    return torch.norm(x, dim=-1, keepdim=True)

def maximum(x, y, *z):
    y = y if isinstance(y, torch.Tensor) else full(x, y)
    m = torch.max(x, y)
    return maximum(m, *z) if z else m


def minimum(x, y, *z):
    y = y if isinstance(y, torch.Tensor) else full(x, y)
    m = torch.min(x, y)
    return minimum(m, *z) if z else m


def clamp(x, a, b):
    return minimum(maximum(x, a), b)


def mix(x, y, a):
    return x * (1.0 - a) + y * a


def stack(*x):
    return torch.stack(x, dim=-2)


def unstack(x):
    return torch.unbind(x, dim=-2)


def concat(*x):
    return torch.cat(x, dim=-1)


def unconcat(x):
    return torch.unbind(unsqueeze(x), dim=-2)


def squeeze(x):
    return torch.squeeze(x, dim=-1)


def unsqueeze(x):
    return torch.unsqueeze(x, dim=-1)


def matmul(A, x):
    return squeeze(A @ unsqueeze(x))


def dot(x, y):
    return torch.sum(x * y, dim=-1, keepdim=True)


def ndot(x, y):
    x, y = unconcat(x * y)
    return x - y


def transpose(A):
    return torch.transpose(A, -2, -1)


def abs(x):
    return torch.abs(x)


def sqrt(x):
    return torch.sqrt(x + 1e-6)


def sign(x):
    return torch.sign(x)


def cos(x):
    return torch.cos(x)


def sin(x):
    return torch.sin(x)


def relu(x):
    return nn.functional.relu(x)


def mod(x, y):
    return torch.fmod(x, y)


def round(x):
    return torch.round(x)


def zero(x):
    return torch.zeros_like(x)


def one(x):
    return torch.ones_like(x)


def full(x, y):
    return torch.full_like(x, y)


def expand(x, y):
    while x.dim() < y.dim():
        x = torch.unsqueeze(x, dim=0)
    x = x.expand_as(y)
    return x

def tensor(x, y):
    return x.new_tensor(y)


def union(sdf1, sdf2):
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        d = minimum(d1, d2)
        return d
    return wrapper


def subtraction(sdf1, sdf2):
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        d = maximum(-d1, d2)
        return d
    return wrapper


def intersection(sdf1, sdf2):
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        d = maximum(d1, d2)
        return d
    return wrapper


def smooth_union(sdf1, sdf2, k):
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
        d = mix(d2, d1, h) - k * h * (1.0 - h)
        return d
    return wrapper


def smooth_subtraction(sdf1, sdf2, k):
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        h = torch.clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
        d = torch.mix(d2, -d1, h) + k * h * (1.0 - h)
        return d
    return wrapper


def smooth_intersection(sdf1, sdf2, k):
    def wrapper(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        h = torch.clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
        d = torch.mix(d2, d1, h) + k * h * (1.0 - h)
        return d
    return wrapper


def translation(sdf, t):
    def wrapper(p):
        d = sdf(p - t)
        return d

    return wrapper


def rotation(sdf, R):
    def wrapper(p):
        d = sdf(matmul(transpose(R), p))
        return d

    return wrapper


def scaling(sdf, s):
    def wrapper(p):
        d = sdf(p / s) * s
        return d

    return wrapper


def elongation(sdf, s):
    def wrapper(p):
        q = abs(p) - s
        d = sdf(relu(q)) - relu(-maximum(*unconcat(q)))
        return d

    return wrapper


def rounding(sdf, r):
    def wrapper(p):
        d = sdf(p) - r
        return d

    return wrapper


def onion(sdf, t):
    def wrapper(p):
        d = abs(sdf(p)) - t
        return d

    return wrapper


def infinite_repetition(sdf, c):
    def wrapper(p):
        q = mod(p + 0.5 * c, c) - 0.5 * c
        d = sdf(q)
        return d

    return wrapper


def finite_repetition(sdf, c, l):
    def wrapper(p):
        q = p - c * clamp(round(p / c), -l, l)
        d = sdf(q)
        return d

    return wrapper


def twist(sdf, k):
    def wrapper(p):
        px, py, pz = unconcat(p)
        c = cos(k * py)
        s = sin(k * py)
        m = stack(concat(c, -s), concat(s, c))
        p = concat(px, pz)
        q = concat(matmul(m, p), py)
        d = sdf(q)
        return d

    return wrapper


def bend(sdf, k):
    def wrapper(p):
        px, py, pz = unconcat(p)
        c = cos(k * px)
        s = sin(k * px)
        m = stack(concat(c, -s), concat(s, c))
        q = concat(matmul(m, concat(px, py)), pz)
        d = sdf(q)
        return d

    return wrapper


def symmetry_x(sdf):
    def wrapper(p):
        px, py, pz = unconcat(p)
        d = sdf(concat(abs(px), py, pz))
        return d

    return wrapper


def symmetry_y(sdf):
    def wrapper(p):
        px, py, pz = unconcat(p)
        d = sdf(concat(px, abs(py), pz))
        return d

    return wrapper


def symmetry_z(sdf):
    def wrapper(p):
        px, py, pz = unconcat(p)
        d = sdf(concat(px, py, abs(pz)))
        return d

    return wrapper

""" 
SDF: Signed Distance Functions
Re-implementations of SDFs based on the following Inigo Quilez's excellent article.
https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
"""
def sphere_sdf(r):
    def sdf(p):
        d = length(p) - r
        return d

    return sdf


def box_sdf(s):
    def sdf(p):
        q = abs(p) - s
        d = length(relu(q)) - relu(-maximum(*unconcat(q)))
        return d

    return sdf


def torus_sdf(r1, r2):
    def sdf(p):
        px, py, pz = unconcat(p)
        q = concat(length(concat(px, pz)) - r1, py)
        d = length(q) - r2
        return d

    return sdf


def link_sdf(l, r1, r2):
    def sdf(p):
        px, py, pz = unconcat(p)
        q = concat(px, relu(abs(py) - l), pz)
        qx, qy, qz = unconcat(q)
        d = length(concat(length(concat(qx, qy)) - r1, qz)) - r2
        return d

    return sdf


def cylinder_sdf(r, h):
    def sdf(p):
        px, py, pz = unconcat(p)
        d = abs(concat(length(concat(px, pz)), py)) - concat(r, h)
        d = -relu(-maximum(*unconcat(d))) + length(relu(d))
        return d

    return sdf


def cone_sdf(r, h):
    def sdf(p):
        # cx, cy = unconcat(c)
        # q = h * concat(cx / cy, -one(cx))
        q = concat(r, -h)
        px, py, pz = unconcat(p)
        qx, qy = unconcat(q)
        w = concat(length(concat(px, pz)), py)
        wx, wy = unconcat(w)
        a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0)
        b = w - q * concat(clamp(wx / qx, 0.0, 1.0), one(wx))
        k = sign(qy)
        d = minimum(dot(a, a), dot(b, b))
        s = maximum(k * (wx * qy - wy * qx), k * (wy - qy))
        d = sqrt(d) * sign(s)
        return d

    return sdf


def capsule_sdf(r, h):
    def sdf(p):
        px, py, pz = unconcat(p)
        py = py - clamp(py, 0.0, h)
        d = length(concat(px, py, pz)) - r
        return d

    return sdf


def ellipsoid_sdf(r):
    def sdf(p):
        k1 = length(p / r)
        d = (k1 - 1.0) * minimum(*unconcat(r))
        return d

    return sdf


def rhombus_sdf(s, h):
    def sdf(p):
        p = abs(p)
        px, py, pz = unconcat(p)
        sx, sy = unconcat(s)
        f = clamp(ndot(s, s - 2.0 * concat(px, pz)) / dot(s, s), -1.0, 1.0)
        q = concat(length(concat(px, pz) - 0.5 * s * concat(1.0 - f, 1.0 + f)) * sign(px * sy + pz * sx - sx * sy),
                   py - h)
        return -relu(-maximum(*unconcat(q))) + length(relu(q))

    return sdf


def triprism_sdf(s, h):
    def sdf(p):
        q = abs(p)
        px, py, pz = unconcat(p)
        qx, qy, qz = unconcat(q)
        d = maximum(qz - h, maximum(qx * np.cos(np.pi / 6.0) + py * np.sin(np.pi / 6.0), -py) - s * 0.5)
        return d

    return sdf


def hexprism_sdf(s, h):
    def sdf(p):
        k = tensor(p, [-np.cos(np.pi / 6.0), np.sin(np.pi / 6.0), np.tan(np.pi / 6.0)])
        kx, ky, kz = unconcat(k)
        q = abs(p)
        qx, qy, qz = unconcat(q)
        t = 2.0 * relu(-dot(concat(kx, ky), concat(qx, qy)))
        qx = qx + t * kx
        qy = qy + t * ky
        d = concat(length(concat(qx, qy) - concat(clamp(qx, -kz * s, kz * s), expand(s, qx))) * sign(qy - s), qz - h)
        d = -relu(-maximum(*unconcat(d))) + length(relu(d))
        return d

    return sdf


def octahedron_sdf(s):
    def sdf(p):
        p = abs(p)
        px, py, pz = unconcat(p)
        m = px + py + pz - s
        q = torch.where(
            3.0 * px < m,
            concat(px, py, pz),
            torch.where(
                3.0 * py < m,
                concat(py, pz, px),
                concat(pz, px, py),
            ),
        )
        qx, qy, qz = unconcat(q)
        k = clamp(0.5 * (qz - qy + s), 0.0, s);
        d = length(concat(qx, qy - s + k, qz - k))
        d = torch.where((3.0 * px < m) | (3.0 * py < m) | (3.0 * pz < m), d, m * np.tan(np.pi / 6.0))
        return d

    return sdf


def pyramid_sdf(h):
    def sdf(p):
        m = h ** 2 + 0.25
        px, py, pz = unconcat(p)
        px, pz = abs(px), abs(pz)
        px, pz = maximum(px, pz), minimum(pz, px)
        px, pz = px - 0.5, pz - 0.5
        q = concat(pz, h * py - 0.5 * px, h * px + 0.5 * py)
        qx, qy, qz = unconcat(q)
        s = relu(-qx)
        t = clamp((qy - 0.5 * pz) / (m + 0.25), 0.0, 1.0)
        a = m * (qx + s) ** 2 + qy ** 2
        b = m * (qx + 0.5 * t) ** 2 + (qy - m * t) ** 2
        d = torch.where(minimum(qy, -qx * m - qy * 0.5) > 0.0, zero(a), minimum(a, b))
        d = sqrt((d + qz ** 2) / m) * sign(maximum(qz, -py))
        return d

    return sdf


def plane_sdf(n, h):
    def sdf(p):
        d = dot(p, n) + h
        return d

    return sdf


def mandelbulb_sdf(power=8, iters=10):
    def sdf(orig_p):
        dr = torch.ones_like(orig_p[:, :, 0:1]).to(orig_p.device) * 2.0
        tmask = torch.ones_like(dr, dtype=torch.bool).to(orig_p.device)
        sdf = torch.zeros_like(dr, dtype=torch.float32).to(orig_p.device)
        tmp_pos = orig_p
        for tmp_iter in range(iters):
            r = length(tmp_pos)
            cur_mask = torch.logical_and(r > 2.0, tmask)
            sdf[cur_mask] = 0.5 * torch.log(r[cur_mask]) * torch.div(r[cur_mask], dr[cur_mask])
            tmask[r > 2.0] = False
            # if r > 2.0:
            #     break
            ## approximate the distance differential
            dr = power * torch.pow(r, power - 1.0) * dr + 1.0
            ## calculate fractal surface
            ## convert to polar coordinates
            theta = torch.acos(torch.div(tmp_pos[:, :, 2:], r))
            phi = torch.atan2(tmp_pos[:, :, 1:2], tmp_pos[:, :, 0:1])
            zr = torch.pow(r, power)
            ## convert back to cartesian coordinated
            x = zr * sin(theta * power) * cos(phi * power)
            y = zr * sin(theta * power) * sin(phi * power)
            z = zr * cos(theta * power)
            tmp_pos = orig_p + torch.cat((x, y, z), dim=-1)
        ## distance estimator
        # return 0.5 * torch.log(r) * r / dr
        return sdf

    return sdf


def mandelbox_sdf(scale, iters, c):

    def sdf(orig_p):
        # x, y, z = unconcat(p)
        p = torch.clone(orig_p)
        cfactor = torch.ones_like(p).to(p.device) * c
        DEfactor = torch.ones_like(p[:, : , 0:1]).to(p.device) * scale
        fixedRadius = torch.ones(1, dtype=torch.float32).to(p.device)
        fR2 = fixedRadius ** 2
        minRadius = torch.ones(1, dtype=torch.float32).to(p.device) * 0.5
        mR2 = minRadius ** 2
        for i in range(iters):
            p[p > 1.0] = 2.0 - p[p > 1.0]
            p[p < -1.0] = -2.0 - p[p < -1.0]
            r2 = length(p).squeeze()
            p[r2 < mR2] = p[r2 < mR2] * fR2 / mR2
            DEfactor[r2 < mR2] = DEfactor[r2 < mR2] * fR2 / mR2
            p[r2 < fR2] = p[r2 < fR2] * fR2 / r2[r2 < fR2][:, None]
            DEfactor[r2 < fR2] = DEfactor[r2 < fR2] * fR2 / r2[r2 < fR2][:, None]
            p = p * scale + cfactor
            DEfactor = DEfactor * scale
        return length(p) / abs(DEfactor)

    return sdf
