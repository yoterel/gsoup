import numpy as np
from gsoup.viewer import gviewer
from gsoup import (
    structures,
    pixels_in_world_space,
    to_hom,
    ray_ray_intersection,
    swap_columns,
    reconstruct_pointcloud,
    load_image,
)

"""
a few example functions for using the viewer
"""

#### globals
ui_float = 0.0
poses = None
meshes_v = None
meshes_f = None
gv_scalar, gv_color, gv_vector, gf_scalar, gf_color = None, None, None, None, None
pcs_v = None
#### globals


def pcs_slider_callback():
    global ui_float, pcs_v, gv_scalar, gv_color
    changed, ui_float = gviewer.psim.SliderFloat(
        "step", ui_float, v_min=0, v_max=len(pcs_v)
    )
    if changed:
        if int(ui_float) >= len(pcs_v):
            ui_float = len(pcs_v) - 1
        if gv_scalar[0] is not None:
            gviewer.register_pointcloud(
                "pc", pcs_v[int(ui_float)], s=gv_scalar[int(ui_float)], radius=0.0006
            )
        elif gv_color[0] is not None:
            gviewer.register_pointcloud(
                "pc", pcs_v[int(ui_float)], c=gv_color[int(ui_float)], radius=0.0006
            )
        else:
            gviewer.register_pointcloud("pc", pcs_v[int(ui_float)], radius=0.0006)


def meshes_slider_callback():
    global ui_float, meshes_v, meshes_f, gv_scalar, gv_color, gv_vector, gf_scalar, gf_color
    changed, ui_float = gviewer.psim.SliderFloat(
        "step", ui_float, v_min=0, v_max=len(meshes_v)
    )
    if changed:
        if int(ui_float) >= len(meshes_v):
            ui_float = len(meshes_v) - 1
        float1, float2, float3, float4, float5 = (
            ui_float,
            ui_float,
            ui_float,
            ui_float,
            ui_float,
        )
        if gv_scalar[0] is None:
            float1 = 0
        if gv_color[0] is None:
            float2 = 0
        if gv_vector[0] is None:
            float3 = 0
        if gf_scalar[0] is None:
            float4 = 0
        if gf_color[0] is None:
            float5 = 0
        gviewer.register_mesh(
            "mesh",
            meshes_v[int(ui_float)],
            meshes_f[int(ui_float)],
            s_vertices=gv_scalar[int(float1)],
            c_vertices=gv_color[int(float2)],
            v_vertices=gv_vector[int(float3)],
            s_faces=gf_scalar[int(float4)],
            c_faces=gf_color[int(float5)],
            edge_width=1.0,
        )


def poses_slider_callback():
    global ui_float, poses
    edge_rad = 0.0005
    changed, ui_float = gviewer.psim.SliderFloat(
        "step", ui_float, v_min=0, v_max=len(poses)
    )
    if changed and poses is not None:
        if int(ui_float) >= len(poses):
            ui_float = len(poses) - 1
        v_tot, e_tot, c_tot = gviewer.register_camera(
            "poses", poses[int(ui_float)], edge_rad, group=True
        )


def pcs_slide_view(v, s_vertices=None, c_vertices=None):
    """
    given some point cloud tXVx3
    show the point cloud as it changes through time t and allow scrolling through using a slider.
    """
    global pcs_v, gv_scalar, gv_color
    gviewer.init()
    gviewer.ps.set_up_dir("z_up")
    gviewer.ps.set_ground_plane_mode("none")
    typical_scale = np.linalg.norm(v, axis=-1).max()
    v /= typical_scale[..., None]
    edge_rad = 0.01
    v_aabb, e_aabb, c_aabb = structures.get_aabb_coords()
    aabb_network = gviewer.ps.register_curve_network(
        "aabb", v_aabb, e_aabb, radius=edge_rad
    )
    aabb_network.add_color_quantity("color", c_aabb, defined_on="edges", enabled=True)
    v_gizmo, e_gizmo, c_gizmo = structures.get_gizmo_coords(0.1)
    gizmo_network = gviewer.ps.register_curve_network(
        "gizmo", v_gizmo, e_gizmo, radius=edge_rad
    )
    gizmo_network.add_color_quantity("color", c_gizmo, defined_on="edges", enabled=True)
    pcs_v = v
    if s_vertices is None:
        gv_scalar = [None]
    else:
        gv_scalar = s_vertices
    if c_vertices is None:
        gv_color = [None]
    else:
        gv_color = c_vertices
    gviewer.ps.set_user_callback(pcs_slider_callback)
    gviewer.ps.set_up_dir("z_up")
    gviewer.register_pointcloud(
        "pc", pcs_v[0], s=gv_scalar[0], c=gv_color[0], radius=0.0006
    )
    gviewer.show()


def meshes_slide_view(
    v, f, s_vertices=None, c_vertices=None, v_vertice=None, s_faces=None, c_faces=None
):
    """
    given some vertices tXVx3 and faces tXFx3
    show the mesh as it changes through time t and allow scrolling through using a slider.
    """
    global meshes_v, meshes_f, gv_scalar, gv_color, gv_vector, gf_scalar, gf_color
    gviewer.init()
    meshes_v = v
    meshes_f = f
    ### next hack is very eww
    if s_vertices is None:
        gv_scalar = [None]
    else:
        gv_scalar = s_vertices
    if c_vertices is None:
        gv_color = [None]
    else:
        gv_color = c_vertices
    if v_vertice is None:
        gv_vector = [None]
    else:
        gv_vector = v_vertice
    if s_faces is None:
        gf_scalar = [None]
    else:
        gf_scalar = s_faces
    if c_faces is None:
        gf_color = [None]
    else:
        gf_color = c_faces

    gviewer.ps.set_user_callback(meshes_slider_callback)
    gviewer.ps.set_up_dir("z_up")
    gviewer.register_mesh(
        "mesh",
        meshes_v[0],
        meshes_f[0],
        s_vertices=gv_scalar[0],
        c_vertices=gv_color[0],
        v_vertices=gv_vector[0],
        s_faces=gf_scalar[0],
        c_faces=gf_color[0],
        edge_width=1.0,
    )
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
    poses = camera_poses[2:]
    gviewer.init()
    gviewer.ps.set_user_callback(poses_slider_callback)
    gviewer.ps.set_up_dir("z_up")
    gviewer.ps.set_ground_plane_mode("shadow_only")
    edge_rad = 0.0005
    point_rad = 0.002
    v_aabb, e_aabb, c_aabb = structures.get_aabb_coords()
    aabb_network = gviewer.ps.register_curve_network(
        "aabb", v_aabb, e_aabb, radius=edge_rad
    )
    aabb_network.add_color_quantity("color", c_aabb, defined_on="edges", enabled=True)
    v_gizmo, e_gizmo, c_gizmo = structures.get_gizmo_coords(0.1)
    gizmo_network = gviewer.ps.register_curve_network(
        "gizmo", v_gizmo, e_gizmo, radius=edge_rad
    )
    gizmo_network.add_color_quantity("color", c_gizmo, defined_on="edges", enabled=True)
    coa = np.zeros((1, 3))
    gviewer.register_pointcloud(
        "center_of_world",
        coa,
        c=np.array([1.0, 1.0, 1.0])[None, :],
        radius=0.005,
        mode="sphere",
    )
    v_tot, e_tot, c_tot = gviewer.register_camera(
        "poses_init", camera_poses[0], edge_rad, group=True, alpha=0.3
    )
    v_tot, e_tot, c_tot = gviewer.register_camera(
        "poses_orig", camera_poses[1], edge_rad, group=True, alpha=0.3
    )
    gviewer.show()


def poses_static_view(
    camera_poses=None, meshes=None, pointclouds=None, group_cameras=True
):
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
    aabb_network = gviewer.ps.register_curve_network(
        "aabb", v_aabb, e_aabb, radius=edge_rad
    )
    aabb_network.add_color_quantity("color", c_aabb, defined_on="edges", enabled=True)
    v_gizmo, e_gizmo, c_gizmo = structures.get_gizmo_coords(0.1)
    gizmo_network = gviewer.ps.register_curve_network(
        "gizmo", v_gizmo, e_gizmo, radius=edge_rad
    )
    gizmo_network.add_color_quantity("color", c_gizmo, defined_on="edges", enabled=True)
    coa = np.zeros((1, 3))
    gviewer.register_pointcloud(
        "center_of_world",
        coa,
        c=np.array([1.0, 1.0, 1.0])[None, :],
        radius=0.005,
        mode="sphere",
    )
    if camera_poses is not None:
        v_tot, e_tot, c_tot = gviewer.register_camera(
            "cameras", camera_poses, edge_rad, group_cameras
        )
    if meshes is not None:
        for i, mesh in enumerate(meshes):
            gviewer.register_mesh(
                "mesh_{}".format(i), mesh[0], mesh[1], transparency=0.5
            )
    if pointclouds is not None:
        typical_scale = np.linalg.norm(pointclouds, axis=-1).max()
        for i, pointcloud in enumerate(pointclouds):
            pointcloud /= typical_scale[..., None]
            gviewer.register_pointcloud("pc_{}".format(i), pointcloud, radius=1e-4)
    gviewer.show()


def calibration_static_view(
    camera_pose,
    projector_pose,
    camera_wh,
    projector_wh,
    camera_intrinsics=None,
    camera_distortion=None,
    projector_intrinsics=None,
    forward_map=None,
    fg=None,
    RGB_color_image=None,
    mode="xy",
):
    """
    visualizes a procam pair reconstruction
    :param camera_pose: (4, 4) np array of camera to world transform
    :param projector_pose: (4, 4) np array of projector to world transform
    :param camera_wh: (2,) np array of camera resolution
    :param projector_wh: (2,) np array of projector resolution
    :param camera_intrinsics: (3, 3) np array of camera intrinsics
    :param camera_distortion: (5,) np array of camera distortion
    :param projector_intrinsics: (3, 3) np array of projector intrinsics
    :param forward_map: (h, w, 2) np array of projector to camera mapping
    :param fg: (h, w, 3) np array of projector fg
    :param mode: "xy" or "ij" for forward map last channel encoding
    """
    gviewer.init(height=1024, width=1024)
    gviewer.ps.set_up_dir("z_up")
    edge_rad = 0.0005
    typical_scale = (
        np.linalg.norm(camera_pose[:-1, :]) + np.linalg.norm(projector_pose[:-1, :])
    ) / 2
    camera_pose[:-1, :] /= typical_scale
    projector_pose[:-1, :] /= typical_scale
    v_aabb, e_aabb, c_aabb = structures.get_aabb_coords()
    aabb_network = gviewer.ps.register_curve_network(
        "aabb", v_aabb, e_aabb, radius=edge_rad
    )
    aabb_network.add_color_quantity("color", c_aabb, defined_on="edges", enabled=True)
    v_gizmo, e_gizmo, c_gizmo = structures.get_gizmo_coords(0.1)
    gizmo_network = gviewer.ps.register_curve_network(
        "gizmo", v_gizmo, e_gizmo, radius=edge_rad
    )
    gizmo_network.add_color_quantity("color", c_gizmo, defined_on="edges", enabled=True)
    coa = np.zeros((1, 3))
    gviewer.register_pointcloud(
        "center_of_world",
        coa,
        c=np.array([1.0, 1.0, 1.0])[None, :],
        radius=0.005,
        mode="sphere",
    )
    v_tot, e_tot, c_tot = gviewer.register_camera(
        "camera", camera_pose[None, ...], edge_rad, True, 1.0, 10.0
    )
    v_tot, e_tot, c_tot = gviewer.register_camera(
        "projector", projector_pose[None, ...], edge_rad, True, 1.0, 10.0
    )
    camera_pixels = pixels_in_world_space(camera_wh, camera_intrinsics, camera_pose)
    projector_pixels = pixels_in_world_space(
        projector_wh, projector_intrinsics, projector_pose
    )
    # camera_screen_pc = gviewer.register_pointcloud("camera_screen", camera_pixels, radius=0.0002)
    # projector_screen_pc = gviewer.register_pointcloud("projector_screen", projector_pixels, radius=0.0002)
    (
        points,
        weight_factor,
        cam_origins,
        cam_directions,
        projector_origins,
        projector_directions,
        cam_pixels,
        projector_pixels,
    ) = reconstruct_pointcloud(
        forward_map,
        fg,
        camera_pose,
        projector_pose,
        camera_intrinsics,
        camera_distortion,
        projector_intrinsics,
        mode=mode,
        color_image=RGB_color_image,
        debug=True,
    )
    reconst = gviewer.register_pointcloud(
        "reconstruction", points[:, :3], c=points[:, 3:], radius=0.0002
    )
    t = np.linspace(0, 3, 50)
    cam_ray_points = cam_origins[0] + t[:, None] * cam_directions[0][None, :]
    cam_ray = gviewer.ps.register_curve_network(
        "cam_ray", cam_ray_points, "line", radius=0.0002
    )
    proj_ray_points = (
        projector_origins[0] + t[:, None] * projector_directions[0][None, :]
    )
    proj_ray = gviewer.ps.register_curve_network(
        "proj_ray", proj_ray_points, "line", radius=0.0002
    )
    ray_intersection = points[0:1, :3]
    gviewer.register_pointcloud(
        "ray_intersection",
        ray_intersection,
        c=np.array([1.0, 1.0, 1.0])[None, :],
        radius=0.0005,
        mode="sphere",
    )
    ###
    gviewer.show()
