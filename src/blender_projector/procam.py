import bpy
from pathlib import Path
import json
import numpy as np


def get_c2w_opencv(camera):
    """
    converts a c2w matrix of blender, into opencv c2w (x is right, y is down, z goes into view direction)
    """
    R_bcam2cv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    c2w = np.matmul(np.array(camera.matrix_world), R_bcam2cv)
    return c2w


def get_scene_resolution(scene):
    resolution_scale = scene.render.resolution_percentage / 100.0
    resolution_x = scene.render.resolution_x * resolution_scale  # [pixels]
    resolution_y = scene.render.resolution_y * resolution_scale  # [pixels]
    return int(resolution_x), int(resolution_y)


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == "VERTICAL":
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == "AUTO":
        if size_x >= size_y:
            return "HORIZONTAL"
        else:
            return "VERTICAL"
    return sensor_fit


def get_camera_parameters_intrinsic(camera):
    """Get intrinsic camera parameters: focal length and principal point."""
    # ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera/120063#120063
    scene = bpy.context.scene
    focal_length = camera.data.lens  # [mm]
    res_x, res_y = get_scene_resolution(scene)
    cam_data = camera.data
    sensor_size_in_mm = get_sensor_size(
        cam_data.sensor_fit, cam_data.sensor_width, cam_data.sensor_height
    )
    sensor_fit = get_sensor_fit(
        cam_data.sensor_fit,
        scene.render.pixel_aspect_x * res_x,
        scene.render.pixel_aspect_y * res_y,
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == "HORIZONTAL":
        view_fac_in_px = res_x
    else:
        view_fac_in_px = pixel_aspect_ratio * res_y
    pixel_size_mm_per_px = (sensor_size_in_mm / focal_length) / view_fac_in_px
    skew = 0
    f_x = 1.0 / pixel_size_mm_per_px
    f_y = (1.0 / pixel_size_mm_per_px) / pixel_aspect_ratio
    c_x = (res_x) / 2.0 - cam_data.shift_x * view_fac_in_px
    c_y = (res_y) / 2.0 + (cam_data.shift_y * view_fac_in_px) / pixel_aspect_ratio
    K = np.array([[f_x, skew, c_x], [0, f_y, c_y], [0, 0, 1]])
    return K


def save_cameras(dst_path):
    """
    saves all cameras' extrinsics and intrinsics to json file
    """
    scene = bpy.context.scene
    res_x, res_y = get_scene_resolution(scene)
    projector_lamp = scene.objects["projector_lamp"]
    proj_resx = (
        projector_lamp.data.node_tree.nodes["proj_resx"].outputs[0].default_value
    )
    proj_resy = (
        projector_lamp.data.node_tree.nodes["proj_resy"].outputs[0].default_value
    )
    cams = {}
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            ext = get_c2w_opencv(obj)
            intr = get_camera_parameters_intrinsic(obj)
            if "projector" in obj.name:
                intr[0, :] /= res_x
                intr[0, :] *= proj_resx
                intr[1, :] /= res_y
                intr[1, :] *= proj_resy
                rx = proj_resx
                ry = proj_resy
            else:
                rx = res_x
                ry = res_y
            # cam_obj = obj          # Object
            # cam_data = obj.data    # Camera datablock
            cams[obj.name] = {
                "extrinsics": ext.tolist(),
                "intrinsics": intr.tolist(),
                "resx": rx,
                "resy": ry,
            }
    with open(Path(dst_path, "cameras.json"), "w") as out_file:
        json.dump(cams, out_file, indent=4)


def update_projector(resx=None, resy=None, vignette=None):
    """
    updates the state of the projector
    Note: used to equalize the underlying camera and shader data. camera parameters get prioritized.
    """
    scene = bpy.context.scene
    projector = scene.objects["projector"]
    projector_camera = scene.objects["projector_camera"]
    projector_lamp = scene.objects["projector_lamp"]
    projector_camera_data = projector_camera.data
    focal_length = projector_camera_data.lens  # [mm]
    shift_x = projector_camera_data.shift_x
    shift_y = projector_camera_data.shift_y
    sensor_size_in_mm = get_sensor_size(
        projector_camera_data.sensor_fit,
        projector_camera_data.sensor_width,
        projector_camera_data.sensor_height,
    )
    if resx is not None:
        projector_lamp.data.node_tree.nodes["proj_resx"].outputs[0].default_value = resx
    if resy is not None:
        projector_lamp.data.node_tree.nodes["proj_resy"].outputs[0].default_value = resy
    if vignette is not None:
        projector_lamp.data.node_tree.nodes["projector"].inputs[
            "Vignette"
        ].default_value = vignette
    projector_lamp.data.node_tree.nodes["focal_length"].outputs[
        0
    ].default_value = focal_length
    projector_lamp.data.node_tree.nodes["lense_shift_x"].outputs[
        0
    ].default_value = shift_x
    projector_lamp.data.node_tree.nodes["lense_shift_y"].outputs[
        0
    ].default_value = shift_y
    projector_lamp.data.node_tree.nodes["sensor_size"].outputs[
        0
    ].default_value = sensor_size_in_mm


def remove_textures():
    # remove all textures from scene
    exceptions = ["uvgrid", "all_white_tex"]
    for image in bpy.data.images:
        if image.name not in exceptions:
            print("removing texture: {}".format(image))
            bpy.data.images.remove(image)


def remove_materials():
    # removes all materials from scene
    exceptions = ["diffmat", "flat_white", "flat_black", "backplane_material"]
    for material in bpy.data.materials:
        if material.name not in exceptions:
            print("removing material: {}".format(material))
            bpy.data.materials.remove(material)


def get_constant_objects():
    constant_objects = [
        "camera_container",
        "camera",
        "projector_container",
        "projector",
        "projector_camera",
        "projector_lamp",
        "backplane",
        "plane_1",
        "plane_2",
        "debug_target",
        "fixed_plane",
        "fixed_area",
    ]
    return constant_objects


def remove_objects():
    for obj in bpy.data.objects:
        if obj.name not in get_constant_objects():
            bpy.data.objects.remove(obj)


def remove_nodes_and_links():
    # compositor nodes
    tree = bpy.context.scene.node_tree
    for node in tree.nodes:
        tree.nodes.remove(node)
    # world links
    world_tree = bpy.data.worlds[0].node_tree
    for link in world_tree.links:
        if link.to_node.name == "Background":
            world_tree.links.remove(link)
