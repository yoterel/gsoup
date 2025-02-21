import numpy as np
from .core import to_hom, to_44
from .transforms import invert_rigid
from .geometry_basic import triangulate_quad_mesh


def barycentric(p, a, b, c):
    """
    get 2D barycentric coordinates of 2D point p for triangle defined by endpoints a,b,c
    :param p: (2,) np.float32 coordinates of point in same coordinate system as a,b,c
    :param a,b,c: (3,) np.float32 end points of triangle
    :return: barycentric coordinates u,v,w of p
    """
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def is_inside_triangle(p, a, b, c):
    """
    tests if p is inside triangle defined by endpoints a,b,c
    :return: True if p is inside, and the barycentric coordinates of p
    """
    u, v, w = barycentric(p, a, b, c)
    return u >= 0 and v >= 0 and w >= 0, u, v, w


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
    projected /= projected[:, 2:]  # Normalize by depth
    return projected  # second dim is (x, y, depth)


def should_cull_tri(v0, v1, v2, camera_pos):
    """
    a slightly naive cull procedure (should use camera view direction)
    """
    normal = np.cross(v1 - v0, v2 - v0)
    view_dir = v0 - camera_pos
    return np.dot(normal, view_dir) >= 0  # Cull back-facing


def draw_line(image, p0, p1, color):
    """
    Draws a line between p0 and p1 on the image using Bresenham's algorithm.

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
    for y in range(int(min_y), int(max_y + 1)):
        for x in range(int(min_x), int(max_x + 1)):
            if 0 <= x < width and 0 <= y < height:
                inside, u, v, w = is_inside_triangle(
                    np.array([x, y]), a[:2], b[:2], c[:2]
                )
                if inside:
                    depth = u * a[2] + v * b[2] + w * c[2]
                    if depth < depth_buffer[y, x]:
                        depth_buffer[y, x] = depth
                        image[y, x] = color


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
    for f in F:
        if wireframe_occlude:
            v0, v1, v2 = V[f[0]], V[f[1]], V[f[2]]
            camera_pos = invert_rigid(to_44(w2c)[None, :])[0, :3, -1]  # get cam pose
            if should_cull_tri(v0, v1, v2, camera_pos):
                continue
        # Draw an edge from each vertex to the next, wrapping around.
        n = len(f)
        for i in range(n):
            p0 = projected_vertices[f[i]]
            p1 = projected_vertices[f[(i + 1) % n]]
            draw_line(image, p0, p1, color[i])


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
            color = np.tile(color[None, :], len(F))
        for i, f in enumerate(F):
            v0, v1, v2 = V[f[0]], V[f[1]], V[f[2]]
            camera_pos = invert_rigid(to_44(w2c)[None, :])[0, :3, -1]  # get cam pose
            if not should_cull_tri(v0, v1, v2, camera_pos):
                draw_triangle(
                    image,
                    depth_buffer,
                    projected_vertices[f[0]],
                    projected_vertices[f[1]],
                    projected_vertices[f[2]],
                    color[i],
                )
    return image
