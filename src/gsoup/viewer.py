import numpy as np
from . import structures
from .core import to_hom, homogenize


class PolycopeSubStub:

    def add_color_quantity(self, *args, **kwargs):
        return None

    def add_scalar_quantity(self, *args, **kwargs):
        return None

    def add_vector_quantity(self, *args, **kwargs):
        return None


class PolyscopeStub:
    def __init__(self):
        self.ps_net = PolycopeSubStub()

    def init(self):
        return None

    def set_up_dir(self, *kwargs):
        return None

    def register_curve_network(self, *args, **kwargs):
        return PolycopeSubStub()

    def show(self):
        return None

    def register_point_cloud(self, *args, **kwargs):
        return PolycopeSubStub()

try:
    import polyscope as ps
    ps.init()
except RuntimeError:
    ps = PolyscopeStub()


def register_pointcloud(ps, name, points, c=None, s=None, v=None, radius=1e-2, mode="quad"):
    """
    register a point cloud to polyscope
    """
    ps_pointcloud = ps.register_point_cloud(name, points,
                                            radius=radius,
                                            point_render_mode=mode)  # color=np.array([0.5, 0.1, 0.3])  # sphere / quad
    if c is not None:
        ps_pointcloud.add_color_quantity("colors", c, enabled=True)
    if s is not None:
        s = (s - np.min(s)) / (np.max(s) - np.min(s))
        ps_pointcloud.add_scalar_quantity("scalar_value", s,
                                          enabled=True,
                                          vminmax=(0., 1.),
                                          cmap="reds")
    if v is not None:
        # v /= (np.linalg.norm(v, axis=-1) + 1e-6)
        ps_pointcloud.add_vector_quantity("vecs", v,
                                          enabled=True,
                                          radius=0.002,
                                          length=0.1,
                                          color=(0.2, 0.5, 0.5))
    return ps_pointcloud

def register_camera(ps, name, poses, edge_rad, group=True):
    """
    register a camera structure to polyscope
    """
    v_cam, e_cam, c_cam = structures.get_camera_coords()
    v_tot = []
    e_tot = []
    c_tot = []
    for i, pose in enumerate(poses):
        v = homogenize((pose @ to_hom(v_cam).T).T)
        if group:
            v_tot.append(v)
            c_tot.append(c_cam)
            e_tot.append(e_cam + (i * len(v_cam)))
        else:
            ps_net = ps.register_curve_network("{}_{}".format(name, i), v, e_cam, radius=edge_rad)
            ps_net.add_color_quantity("color", c_cam, defined_on='edges', enabled=True)
    if group:
        v_tot = np.array(v_tot).reshape(-1, 3)
        e_tot = np.array(e_tot).reshape(-1, 2)
        c_tot = np.array(c_tot).reshape(-1, 3)
        ps_net = ps.register_curve_network(name, v_tot, e_tot, radius=edge_rad)
        ps_net.add_color_quantity("color", c_tot, defined_on='edges', enabled=True)
    return v_tot, e_tot, c_tot

def view(camera_poses=None, group_cameras=True):
    """
    visualizes a camera setup
    :param camera_pose: (n, 4, 4) np array of camera to world transforms
    :param group_cameras: if true, accelerates view but groups camera as a single object
    :return:
    """
    ps.init()
    ps.set_up_dir("z_up")
    edge_rad = 0.0005
    v_aabb, e_aabb, c_aabb = structures.get_aabb_coords()
    ps_net = ps.register_curve_network("aabb", v_aabb, e_aabb, radius=edge_rad)
    ps_net.add_color_quantity("color", c_aabb, defined_on='edges', enabled=True)
    register_pointcloud(ps, "center_of_world", np.zeros((1, 3)), c=np.array([1., 1., 1.])[None, :], radius=0.005, mode="sphere")
    if camera_poses is not None:
        v_tot, e_tot, c_tot = register_camera(ps, "cameras", camera_poses, edge_rad, group_cameras)
    ps.show()


