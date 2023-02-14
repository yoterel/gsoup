import numpy as np
from gsoup.viewer import gviewer
from gsoup import structures

"""
a few example functions for using the viewer
"""

#### globals 
ui_float = 0.0
poses = None
meshes_v = None
meshes_f = None
meshes_attribute = None
pcs_v = None
pcs_attribute = None
#### globals

def pcs_slider_callback():
    global ui_float, pcs_v
    changed, ui_float = gviewer.psim.SliderFloat("step", ui_float, v_min=0, v_max=len(pcs_v))
    if changed:
        if int(ui_float) >= len(pcs_v):
            ui_float = len(pcs_v)-1
        gviewer.register_pointcloud("pc", pcs_v[int(ui_float)],
                                    s=pcs_attribute[int(ui_float)], radius=0.0006)

def meshes_slider_callback():
    global ui_float, meshes_v, meshes_f
    changed, ui_float = gviewer.psim.SliderFloat("step", ui_float, v_min=0, v_max=len(meshes_v))
    if changed:
        if int(ui_float) >= len(meshes_v):
            ui_float = len(meshes_v)-1
        gviewer.register_mesh("mesh", meshes_v[int(ui_float)],
                    meshes_f[int(ui_float)],
                    v_vertices=meshes_attribute[int(ui_float)], edge_width=1.0)

def poses_slider_callback():
    global ui_float, poses
    edge_rad = 0.0005
    changed, ui_float = gviewer.psim.SliderFloat("step", ui_float, v_min=0, v_max=len(poses))
    if changed and poses is not None:
        if int(ui_float) >= len(poses):
            ui_float = len(poses)-1
        v_tot, e_tot, c_tot = gviewer.register_camera("poses", poses[int(ui_float)], edge_rad, group=True)

def pcs_slide_view(v, v_attribute):
    """
    given some vertices tXVx3 and faces tXFx3
    show the mesh as it changes through time t and allow scrolling through using a slider.
    """
    global pcs_v, pcs_attribute
    gviewer.init()
    pcs_v = v
    pcs_attribute = v_attribute
    gviewer.ps.set_user_callback(pcs_slider_callback)
    gviewer.ps.set_up_dir("z_up")
    gviewer.register_pointcloud("pc", pcs_v[0], s=pcs_attribute[0], radius=0.0006)
    gviewer.show()

def meshes_slide_view(v, f, v_attribute):
    """
    given some vertices tXVx3 and faces tXFx3
    show the mesh as it changes through time t and allow scrolling through using a slider.
    """
    global meshes_v, meshes_f, meshes_attribute
    gviewer.init()
    meshes_v = v
    meshes_f = f
    meshes_attribute = v_attribute
    gviewer.ps.set_user_callback(meshes_slider_callback)
    gviewer.ps.set_up_dir("z_up")
    gviewer.register_mesh("mesh", meshes_v[0], meshes_f[0],
                v_vertices=meshes_attribute[0], edge_width=1.0)
    gviewer.show()

def poses_slide_view(camera_poses):
    """
    given a tensor of t x b x 4 x 4 camera poses, where t is time axis (or step number), b is batch axis, and 4x4 is the c2w transform matrix,
    show the batch of poses and allow scrolling through the time axis using a slider.
    """
    if camera_poses.ndim != 4:
        raise ValueError("camera_poses must be t x b x 4 x 4")
    if camera_poses.shape[2] != 4 or camera_poses.shape[3] != 4:
        raise ValueError("camera_poses must be t x b x 4 x 4")
    global poses
    poses = camera_poses
    gviewer.init()
    gviewer.ps.set_user_callback(poses_slider_callback)
    gviewer.ps.set_up_dir("z_up")
    gviewer.ps.set_ground_plane_mode("shadow_only")
    edge_rad = 0.0005
    point_rad = 0.002
    v_aabb, e_aabb, c_aabb = structures.get_aabb_coords()
    ps_net = gviewer.ps.register_curve_network("aabb", v_aabb, e_aabb, radius=edge_rad)
    ps_net.add_color_quantity("color", c_aabb, defined_on='edges', enabled=True)
    coa = np.zeros((1, 3))
    gviewer.register_pointcloud("center_of_world", coa, c=np.array([1., 1., 1.])[None, :], radius=0.005, mode="sphere")
    v_tot, e_tot, c_tot = gviewer.register_camera("poses_orig", poses[0], edge_rad, group=True, alpha=0.3)
    gviewer.show()

def poses_static_view(camera_poses=None, meshes=None, pointclouds=None, group_cameras=True):
    """
    visualizes a camera setup
    :param camera_pose: (n, 4, 4) np array of camera to world transforms
    :param meshes: list of (v, f) tuples
    :param pointclouds: list of v
    :param group_cameras: if true, accelerates view but groups camera as a single object
    :return:
    """
    gviewer.init(height=512, width=512)
    gviewer.ps.set_up_dir("z_up")
    edge_rad = 0.0005
    v_aabb, e_aabb, c_aabb = structures.get_aabb_coords()
    ps_net = gviewer.ps.register_curve_network("aabb", v_aabb, e_aabb, radius=edge_rad)
    ps_net.add_color_quantity("color", c_aabb, defined_on='edges', enabled=True)
    gviewer.register_pointcloud("center_of_world", np.zeros((1, 3)), c=np.array([1., 1., 1.])[None, :], radius=0.005, mode="sphere")
    if camera_poses is not None:
        v_tot, e_tot, c_tot = gviewer.register_camera("cameras", camera_poses, edge_rad, group_cameras)
    if meshes is not None:
        for i, mesh in enumerate(meshes):
            gviewer.register_mesh("mesh_{}".format(i), mesh[0], mesh[1], transparency=0.5)
    if pointclouds is not None:
        for i, pointcloud in enumerate(pointclouds):
            gviewer.register_pointcloud("pc_{}".format(i), pointcloud, radius=1e-4)
    gviewer.show()