import numpy as np
from .core import to_hom, to_44
from .transforms import invert_rigid
from .geometry_basic import triangulate_quad_mesh


def barycentric(p, a, b, c):
    """
    get 2D barycentric coordinates of 2D point p for triangle defined by endpoints a,b,c
    :param p: (..., 2) np.float32 coordinates of point in same coordinate system as a,b,c
    :param a,b,c: (3,) np.float32 end points of triangle
    :return: barycentric coordinates (..., 3) u,v,w of p
    """
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-6:
        tmp = np.empty((*p.shape[:-1], 3))
        tmp.fill(np.nan)
        return tmp
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.concatenate((u[..., None], v[..., None], w[..., None]), axis=-1)


def is_inside_triangle(p, a, b, c):
    """
    tests if p is inside triangle defined by vertices a,b,c
    :param p: (..., 2) points xy to test for
    :param a,b,c: (2,) triangle vertices
    :return: mask of size (...,) where true if p is inside, and the barycentric coordinates of p (..., 3)
    """
    uvw = barycentric(p, a, b, c)
    # u >= 0 and v >= 0 and w >= 0
    return (uvw >= 0).all(axis=-1), uvw


def project_points(points, K, w2c):
    """
    projects 3D points to camera screen space
    :param points: (n, 3) np.float32 points to be projected
    :param K: (3, 3) np.float32 intrinsics of camera
    :param w2c: (3, 4) np.float32 extrinsics of camera
    """
    points_h = to_hom(points)
    projected = K @ w2c @ points_h.T  # Apply camera projection
    projected = projected.T
    projected[:, :2] /= projected[:, 2:]  # Normalize by depth
    return projected  # second dim is (x, y, depth)


def should_cull_tri(v_in_f, camera_pos):
    """
    a slightly naive triangle cull procedure (should use camera view direction)
    :param v_in_f: (n, 3, 3) coordinates of the vertices of the triangle
    :param camera_pos: (1, 3) camera position
    :return: (n,) mask of triangles to cull
    """
    normal = np.cross(v_in_f[:, 1] - v_in_f[:, 0], v_in_f[:, 2] - v_in_f[:, 0], axis=-1)
    view_dir = v_in_f[:, 0] - camera_pos
    dot_prod = normal[:, None, :] @ view_dir[:, :, None]
    return dot_prod.squeeze() >= 0  # Cull back-facing


def draw_line(image, p0, p1, color):
    """
    draws a line (in-place) between p0 and p1 on the image using Bresenham's algorithm.
    :param image: (height, width, 3) np.uint8 image.
    :param p0: Starting point (x, y).
    :param p1: Ending point (x, y).
    :param color: (3,) np.uint8 color.
    """
    x0, y0 = np.round(p0).astype(np.int32)
    x1, y1 = np.round(p1).astype(np.int32)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < image.shape[1] and 0 <= y0 < image.shape[0]:
            image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def draw_triangle(image, depth_buffer, a, b, c, color):
    """
    draws a triangle a,b,c into image
    """
    height, width = image.shape[:2]
    min_x, max_x = max(0, min(a[0], b[0], c[0])), min(width - 1, max(a[0], b[0], c[0]))
    min_y, max_y = max(0, min(a[1], b[1], c[1])), min(height - 1, max(a[1], b[1], c[1]))
    xx, yy = np.meshgrid(
        np.arange(int(min_x), int(max_x + 1)),
        np.arange(int(min_y), int(max_y + 1)),
        indexing="ij",
    )
    grid = np.concatenate((xx[..., None], yy[..., None]), axis=-1)
    clip = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    inside, uvw = is_inside_triangle(grid, a[:2], b[:2], c[:2])
    depth = uvw[..., 0] * a[2] + uvw[..., 1] * b[2] + uvw[..., 2] * c[2]
    depth_mask = depth < depth_buffer[grid[..., 1], grid[..., 0]]
    draw_mask = clip & inside & depth_mask
    depth_buffer[grid[..., 1][draw_mask], grid[..., 0][draw_mask]] = depth[draw_mask]
    image[grid[..., 1][draw_mask], grid[..., 0][draw_mask]] = color
    ### old serial implementation
    # for y in range(int(min_y), int(max_y + 1)):
    #     for x in range(int(min_x), int(max_x + 1)):
    #         if 0 <= x < width and 0 <= y < height:
    #             inside, u, v, w = is_inside_triangle(
    #                 np.array([x, y]), a[:2], b[:2], c[:2]
    #             )
    #             if inside:
    #                 depth = u * a[2] + v * b[2] + w * c[2]
    #                 if depth < depth_buffer[y, x]:
    #                     depth_buffer[y, x] = depth
    #                     image[y, x] = color


def get_silhouette_edges(V, F, e2f, K, w2c):
    """
    returns a mask (E,) of edges that are silhouette edges, given a borderless mesh
    does not take into account self-occlusions
    :param V: (n, 3) np.float32 vertices of the mesh
    :param F: (m, 3) np.int32 faces of the mesh
    :param e2f: dict mapping edge index to face indices
    :param K: (3, 3) np.float32 camera intrinsics
    :param w2c: (3, 4) np.float32 camera extrinsics
    :return: (E,) np.bool mask of silhouette edges
    """
    camera_pos = invert_rigid(to_44(w2c)[None, :])[:, :3, -1]  # get cam pose (1, 3)
    ##
    verts = V[F[e2f]][
        :, :, :3, :
    ]  # (e, incident_faces (2), verts (3), coordinates (3))
    cull_mask = should_cull_tri(verts.reshape(-1, 3, 3), camera_pos)
    cull_mask = cull_mask.reshape(len(e2f), 2)
    mask = np.logical_xor(cull_mask[:, 0], cull_mask[:, 1])
    # ## old serial implementation
    # mask = np.zeros(len(e2f), dtype=np.bool)
    # for i in range(len(e2f)):
    #     verts = V[F[e2f[i]]][:, :3, :]
    #     v00, v01, v02 = verts[0]
    #     v10, v11, v12 = verts[1]
    #     f0_cull = should_cull_tri(v00, v01, v02, camera_pos)
    #     f1_cull = should_cull_tri(v10, v11, v12, camera_pos)
    #     if (f0_cull and not f1_cull) or (not f0_cull and f1_cull):
    #         mask[i] = True
    return mask


def get_visible_edges(V, F, e2f, K, w2c):
    """
    returns a mask (E,) of edges that are visible (front facing) to camera, given a borderless mesh
    does not take into account self-occlusions
    :param V: (n, 3) np.float32 vertices of the mesh
    :param F: (m, 3) np.int32 faces of the mesh
    :param e2f: dict mapping edge index to face indices
    :param K: (3, 3) np.float32 camera intrinsics
    :param w2c: (3, 4) np.float32 camera extrinsics
    :return: (E,) np.bool mask of silhouette edges
    """
    camera_pos = invert_rigid(to_44(w2c)[None, :])[0, :3, -1]  # get cam pose
    verts = V[F[e2f]][
        :, :, :3, :
    ]  # (e, incident_faces (2), verts (3), coordinates (3))
    cull_mask = should_cull_tri(verts.reshape(-1, 3, 3), camera_pos)
    cull_mask = cull_mask.reshape(len(e2f), 2)
    mask = cull_mask.any(axis=-1)
    # old serial implementation
    # mask = np.zeros(len(e2f), dtype=np.bool)
    # for i in range(len(e2f)):
    #     verts = V[F[e2f[i]]][:, :3, :]
    #     v00, v01, v02 = verts[0]
    #     v10, v11, v12 = verts[1]
    #     f0_cull = should_cull_tri(v00, v01, v02, camera_pos)
    #     f1_cull = should_cull_tri(v10, v11, v12, camera_pos)
    #     if (not f0_cull) or (not f1_cull):
    #         mask[i] = True
    return mask


def render_wireframe(image, V, F, K, w2c, color, wireframe_occlude):
    """
    Renders the wireframe of a mesh. Supports both triangle and quad faces.

    :param image: (height, width, 3) np.uint8 image to draw on.
    :param V: (n, 3) np.float32 vertices in world coordinates.
    :param F: list or array of face indices (each face can be a triangle [3] or a quad [4]).
    :param K: (3, 3) np.float32 camera intrinsics.
    :param w2c: (3, 4) np.float32 camera extrinsics (world -> cam).
    :param color: (m, 3) np.uint8 line color per face.
    :param wireframe_occlude: if true, will not render wireframe on backfaces
    """
    # Project the vertices to screen space.
    projected_vertices = project_points(V, K, w2c)[:, :2]
    camera_pos = invert_rigid(to_44(w2c)[None, :])[0, :3, -1]  # get cam pose
    if wireframe_occlude:
        mask = should_cull_tri(V[F], camera_pos[None, ...])
    else:
        mask = np.ones(len(F), dtype=bool)
    for i, f in enumerate(F):
        if mask[i]:
            continue
        # Draw an edge from each vertex to the next, wrapping around.
        n = len(f)
        for ii in range(n):
            p0 = projected_vertices[f[ii]]
            p1 = projected_vertices[f[(ii + 1) % n]]
            draw_line(image, p0, p1, color[ii])


def render_mesh(
    V,
    F,
    K,
    w2c,
    color,
    wireframe=False,
    wireframe_occlude=False,
    image=None,
    depth_buffer=None,
    wh=(512, 512),
):
    """
    renders a mesh onto an image
    :param V: a (n, 3) np.float32 of vertices of a mesh in world coordinates
    :param F: a (m, 3) or (m, 4) np.int32 indices into V, defining the faces of the mesh (assumes CCW order)
    :param K: a (3, 3) np.float32 intrinsics matrix (opencv convention)
    :param w2c: a (3, 4) np.float32 extrinsics matrix (opencv convention, world -> cam)
    :param color: (m, 3) np.uint8 color per face, or (3,) for single color.
    :param wireframe: if True, will render the wireframe version of the mesh
    :param wireframe_occlude: if True, will not render wireframe on backfaces
    :param image: if not None, a (height, width 3) np.uint8 to be rendered into
    :param depth_buffer: if not None, a (height, width) np.float32 to store depth and perform z-testing
    :param wh: (2-tuple) width height to use if image/depthbuffer aren't provided
    :return (w, h, 3) np.uint8 rendered image
    """
    projected_vertices = project_points(V, K, w2c)  # project vertices to screen space
    if image is None:
        image = np.zeros((wh[1], wh[0], 3), dtype=np.uint8)
    else:
        image = image.copy()
    if depth_buffer is None and not wireframe:
        depth_buffer = np.full((wh[1], wh[0]), np.inf, dtype=np.float32)
    if image.shape[0:2] != image.shape[0:2]:
        raise ValueError("image and depth buffer must have same spatial dimensions")
    if wireframe:
        if color.ndim == 1:  # handle single color
            color = np.tile(color[None, :], (len(F), 1))
        render_wireframe(
            image,
            V,
            F,
            K,
            w2c,
            color,
            wireframe_occlude=wireframe_occlude,
        )
    else:
        if F.shape[-1] == 4:  # triangulate for rendering
            F, color = triangulate_quad_mesh(F, color)
        if color.ndim == 1:  # handle single color
            color = np.tile(color[None, :], (len(F), 1))
        camera_pos = invert_rigid(to_44(w2c)[None, :])[0, :3, -1]  # get cam pose
        mask = should_cull_tri(V[F], camera_pos[None, ...])
        for i, f in enumerate(F):
            if not mask[i]:
                draw_triangle(
                    image,
                    depth_buffer,
                    projected_vertices[f[0]],
                    projected_vertices[f[1]],
                    projected_vertices[f[2]],
                    color[i],
                )
    return image
