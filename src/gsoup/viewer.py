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
        self.SliderFloat = lambda *args, **kwargs: None

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
    
    def register_surface_mesh(self, *args, **kwargs):
        return PolycopeSubStub()

try:
    import polyscope as ps
    import polyscope.imgui as psim
    ps.init()
except RuntimeError:
    ps = PolyscopeStub()
    psim = PolyscopeStub()


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

def register_camera(ps, name, poses, edge_rad, group=True, alpha=1.0):
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
            ps_net.set_transparency(alpha)
    if group:
        v_tot = np.array(v_tot).reshape(-1, 3)
        e_tot = np.array(e_tot).reshape(-1, 2)
        c_tot = np.array(c_tot).reshape(-1, 3)
        ps_net = ps.register_curve_network(name, v_tot, e_tot, radius=edge_rad)
        ps_net.add_color_quantity("color", c_tot, defined_on='edges', enabled=True)
    ps_net.set_transparency(alpha)
    return v_tot, e_tot, c_tot

def register_mesh(ps, name, v, f,
                  transparency=1.0, edge_width=0., color=[0.5, 0.5, 0.5], smooth_shade=True,
                  c_vertices=None, c_faces=None, s_faces=None, v_vertices=None):
    """
    regiter a mesh to polyscope
    :param ps: polyscope instance
    :param name: name of the mesh
    :param v: vertices
    :param f: faces
    :param transparency: transparency of the mesh
    :param edge_width: edge width of the mesh
    :param smooth_shade: smooth shading
    :param c_vertices: vertex scalar values
    :param c_faces: face scalar values
    :param s_faces: face colors
    :param v_vertices: vertex vectors
    """
    ps_mesh = ps.register_surface_mesh(name, v, f,
                                       edge_width=edge_width,
                                       transparency=transparency,
                                       color=color,
                                       smooth_shade=smooth_shade)
    if c_vertices is not None:
        c = (c_vertices - np.min(c_vertices)) / (np.max(c_vertices) - np.min(c_vertices))
        ps_mesh.add_scalar_quantity("vscalar", c,
                                    defined_on="vertices",
                                    enabled=True,
                                    vminmax=(0., 1.),
                                    cmap="rainbow")
    if c_faces is not None:
        c = (c_faces - np.min(c_faces)) / (np.max(c_faces) - np.min(c_faces))
        ps_mesh.add_scalar_quantity("fscalar", c,
                                    defined_on="faces",
                                    enabled=True,
                                    vminmax=(0., 1.),
                                    cmap="rainbow")
    if s_faces is not None:
        ps_mesh.add_color_quantity("fcolor", s_faces,
                                    defined_on="faces",
                                    enabled=True)
    if v_vertices is not None:
        ps_mesh.add_vector_quantity("vecs", v_vertices,
                                    enabled=True,
                                    # vectortype='ambient',
                                    radius=0.01,
                                    length=0.1,
                                    color=(0.2, 0.5, 0.5))
    return ps_mesh

#### global
ui_float = 0.0
poses = None
meshes_v = None
meshes_f = None
meshes_attribute = None
#### global
def meshes_slider_callback():
    global ui_float, meshes_v, meshes_f
    changed, ui_float = psim.SliderFloat("step", ui_float, v_min=0, v_max=len(meshes_v))
    if changed:
        if int(ui_float) >= len(meshes_v):
            ui_float = len(meshes_v)-1
        register_mesh(ps, "mesh", meshes_v[int(ui_float)],
                      meshes_f[int(ui_float)],
                      v_vertices=meshes_attribute[int(ui_float)], edge_width=1.0)

def meshes_slide_view(v, f, v_attribute):
    """
    given a list size t of vertices Vx3 and faces Fx3
    show the mesh as it changes through time t and allow scrolling through using a slider.
    """
    global meshes_v, meshes_f, meshes_attribute
    ps.init()
    meshes_v = v
    meshes_f = f
    meshes_attribute = v_attribute
    ps.set_user_callback(meshes_slider_callback)
    ps.set_up_dir("z_up")
    register_mesh(ps, "mesh", meshes_v[0], meshes_f[0],
                  v_vertices=meshes_attribute[0], edge_width=1.0)
    ps.show()

def poses_slider_callback():
    global ui_float, poses
    edge_rad = 0.0005
    changed, ui_float = psim.SliderFloat("step", ui_float, v_min=0, v_max=len(poses))
    if changed and poses is not None:
        if int(ui_float) >= len(poses):
            ui_float = len(poses)-1
        v_tot, e_tot, c_tot = register_camera(ps, "poses", poses[int(ui_float)], edge_rad, group=True)
        
def poses_slide_view(camera_poses):
    """
    given a tensor of t x b x 4 x 4 camera poses, where t is time axis (or step number), b is batch axis, and 4x4 is the camera pose matrix,
    show the batch of poses and allow scrolling through the time axis using a slider.
    """
    global poses
    poses = camera_poses
    ps.init()
    ps.set_user_callback(poses_slider_callback)
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    edge_rad = 0.0005
    point_rad = 0.002
    v_aabb, e_aabb, c_aabb = structures.get_aabb_coords()
    ps_net = ps.register_curve_network("aabb", v_aabb, e_aabb, radius=edge_rad)
    ps_net.add_color_quantity("color", c_aabb, defined_on='edges', enabled=True)
    register_pointcloud(ps, "center_of_world", np.zeros((1, 3)), c=np.array([1., 1., 1.])[None, :], radius=0.005, mode="sphere")
    v_tot, e_tot, c_tot = register_camera(ps, "poses_orig", poses[0], edge_rad, group=True, alpha=0.3)
    ps.show()

def poses_static_view(camera_poses=None, meshes=None, pointclouds=None, group_cameras=True):
    """
    visualizes a camera setup
    :param camera_pose: (n, 4, 4) np array of camera to world transforms
    :param meshes: list of (v, f) tuples
    :param pointclouds: list of v
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
    if meshes is not None:
        for i, mesh in enumerate(meshes):
            register_mesh(ps, "mesh_{}".format(i), mesh[0], mesh[1], transparency=0.5)
    if pointclouds is not None:
        for i, pointcloud in enumerate(pointclouds):
            register_pointcloud(ps, "pc_{}".format(i), pointcloud, radius=1e-4)
    ps.show()
