import numpy as np
from .core import to_hom, to_44
from .transforms import invert_rigid


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


def project(points, K, Rt):
    """
    projects 3D points to camera screen space
    :param points: (n, 3) np.float32 points to be projected
    :param K: (3, 3) np.float32 intrinsics of camera
    :param Rt: (3, 4) np.float32 extrinsics of camera
    """
    points_h = to_hom(points)
    projected = K @ Rt @ points_h.T  # Apply camera projection
    projected = projected.T
    projected /= projected[:, 2:]  # Normalize by depth
    return projected  # second dim is (x, y, depth)


def should_cull_face(v0, v1, v2, camera_pos):
    """
    a slightly naive cull procedure (should use camera view direction)
    """
    normal = np.cross(v1 - v0, v2 - v0)
    view_dir = v0 - camera_pos
    return np.dot(normal, view_dir) >= 0  # Cull back-facing triangles


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


def render_mesh(image, depth_buffer, V, F, K, Rt, colors):
    """
    renders a mesh using basic rasterization
    :param image: a (height, width 3) np.uint8 to be rendered into
    :param depth_buffer: a (height, width) np.float32 to store depth and perform z-testing
    :param V: a (n, 3) np.float32 of vertices of a mesh in world coordinates
    :param F: a (m, 3) np.int32 indices into V, defining the faces of the mesh
    :param K: a (3, 3) np.float32 intrinsics matrix (opencv convention)
    :param Rt: a (3, 4) np.float32 extrinsics matrix (opencv convention, world -> cam)
    :param colors: (m, 3) np.uint8 color per face.
    """
    projected_vertices = project(V, K, Rt)  # project vertices to screen space
    # extract camera pose by inverting Rt and taking last column
    camera_pos = invert_rigid(to_44(Rt)[None, :])[0, :3, -1]
    for i, f in enumerate(F):
        v0, v1, v2 = V[f[0]], V[f[1]], V[f[2]]
        if not should_cull_face(v0, v1, v2, camera_pos):
            draw_triangle(
                image,
                depth_buffer,
                projected_vertices[f[0]],
                projected_vertices[f[1]],
                projected_vertices[f[2]],
                colors[i],
            )
