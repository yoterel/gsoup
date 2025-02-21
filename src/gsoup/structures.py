import torch
import numpy as np
import torch.nn as nn


def icosehedron(scale=1.0, centered=True):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ]
    ).astype(np.float32)
    if centered:
        vertices -= np.array([0.0, 0.0, 0.0])
    vertices *= scale
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
    ).astype(np.int32)
    return vertices, faces


def quad_cube(scale=1.0, centered=True):
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    if centered:
        vertices -= np.array([0.5, 0.5, 0.5])
    vertices *= scale
    faces = np.array(
        [
            [0, 3, 2, 1],  # Bottom face (when "up" is +Z)
            [1, 6, 5, 0],  # Left face (when looking towards -X)
            [2, 7, 6, 1],  # Front face (when looking towards -X)
            [6, 7, 4, 5],  # Top face (when looking towards -X)
            [0, 5, 4, 3],  # Back face (when looking towards -X)
            [3, 4, 7, 2],  # Right face (when looking towards -X)
        ],
        dtype=np.int32,
    )
    return vertices, faces


def cube(scale=1.0, centered=True):
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    ).astype(np.float32)
    if centered:
        vertices -= np.array([0.5, 0.5, 0.5])
    vertices *= scale
    faces = np.array(
        [
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
        ]
    ).astype(np.int32)
    return vertices, faces


def get_gizmo_coords(scale=1.0):
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ).astype(np.float32)
    vertices *= scale
    edges = np.array([[0, 1], [0, 2], [0, 3]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return vertices, edges, colors


def get_camera_coords(scale=0.1):
    vertices = np.array(
        [
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
        ]
    )
    vertices *= scale
    edges = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [0, 7],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ]
    )
    colors = np.array(
        [
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
        ]
    )
    return vertices, edges, colors


def get_aabb_coords(centered=True):
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    ).astype(np.float32)
    if centered:
        vertices -= np.array([0.5, 0.5, 0.5])
    edges = np.array(
        [
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
        ]
    )
    colors = np.array(
        [
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
        ]
    )
    return vertices, edges, colors


def length(x):
    return torch.norm(x, dim=-1, keepdim=True)


def maximum(x, y, *z):
    y = y if isinstance(y, torch.Tensor) else torch.full_like(x, y)
    m = torch.max(x, y)
    return maximum(m, *z) if z else m


def minimum(x, y, *z):
    y = y if isinstance(y, torch.Tensor) else torch.full_like(x, y)
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


def union(sdf1, sdf2):
    def sdf(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        d = minimum(d1, d2)
        return d

    return sdf


def subtraction(sdf1, sdf2):
    def sdf(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        d = maximum(-d1, d2)
        return d

    return sdf


def intersection(sdf1, sdf2):
    def sdf(p):
        d1 = sdf1(p)
        d2 = sdf2(p)
        d = maximum(d1, d2)
        return d

    return sdf


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
        d = length(nn.functional.relu(q)) - nn.functional.relu(-maximum(*unconcat(q)))
        return d

    return sdf


def torus_sdf(r1, r2):
    def sdf(p):
        px, py, pz = unconcat(p)
        q = concat(length(concat(px, pz)) - r1, py)
        d = length(q) - r2
        return d

    return sdf


def cylinder_sdf(r, h):
    def sdf(p):
        px, py, pz = unconcat(p)
        d = abs(concat(length(concat(px, pz)), py)) - concat(r, h)
        d = -nn.functional.relu(-maximum(*unconcat(d))) + length(nn.functional.relu(d))
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


def plane_sdf(n, h):
    def sdf(p):
        # p dot n + h = 0
        d = torch.sum(p * n, dim=-1, keepdim=True) + h
        return d

    return sdf


def mandelbulb_sdf(power=8, iters=10):
    def sdf(orig_p):
        dr = torch.ones_like(orig_p[:, :, 0:1]).to(orig_p.device) * 2.0
        tmask = torch.ones_like(dr, dtype=torch.bool).to(orig_p.device)
        sdf = torch.zeros_like(dr, dtype=torch.float32).to(orig_p.device)
        tmp_pos = orig_p
        for _ in range(iters):
            r = length(tmp_pos)
            cur_mask = torch.logical_and(r > 2.0, tmask)
            sdf[cur_mask] = (
                0.5 * torch.log(r[cur_mask]) * torch.div(r[cur_mask], dr[cur_mask])
            )
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
            x = zr * torch.sin(theta * power) * torch.cos(phi * power)
            y = zr * torch.sin(theta * power) * torch.sin(phi * power)
            z = zr * torch.cos(theta * power)
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
        DEfactor = torch.ones_like(p[:, :, 0:1]).to(p.device) * scale
        fixedRadius = torch.ones(1, dtype=torch.float32).to(p.device)
        fR2 = fixedRadius**2
        minRadius = torch.ones(1, dtype=torch.float32).to(p.device) * 0.5
        mR2 = minRadius**2
        for _ in range(iters):
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
