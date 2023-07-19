import bpy
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path
import json
import random
import numpy as np
from PIL import Image
import cv2

RESULTS_PATH = 'calibration1'
N_VIEWS = 2  # if >1, will do this number of checkerboard calibration sessions else just graycode on object
RANDOM_TEXTURES = False  # projectors use different patterns each time
RANDOM_VIEWS = False  # views are randomly sampled on a hemisphere
DEBUG = False  # will quit after setting a texture
DEBUG_VIEW_INDEX = 161
MULTI_PROJECTORS = False  # two projectors
AMBIENT_LIGHT = False
COLOC_LIGHT = False
CONSTANT_BG = False
OUTPUT_DEPTH = False
OUTPUT_NORMALS = False
SINGLE_BOUNCE = True
RESOLUTION = 800
PROJ_RES_W = 800
PROJ_RES_H = 800
COLOR_DEPTH = 8
FORMAT = 'PNG'
PROJECTOR_LOC = (0.8, 0.1, 0.2)
CAMERA_LOC = (1.0, 0.0, 0.0)
CHECKERBOARD_LOC = (0.0, 0.0, 0.0)
CAM_DIST_FROM_ZERO = 1
BLENDER_SUFFIX = 161  # eww blender...
normal_layer_name = "Normal"
denoised_layer_name = "Image"
results_dir = Path(bpy.path.abspath(f"//{RESULTS_PATH}"))
patterns_dir = Path(results_dir, "projector")
captures_dir = Path(results_dir, "captures")
####################################################################################
############################### Helper Functions ###################################
####################################################################################

class GrayCode:
    """
    a class that handles encoding and decoding graycode patterns
    """
    def encode1d(self, length):
        total_images = len(bin(length-1)[2:])

        def xn_to_gray(n, x):
            # infer a coordinate gray code from its position x and index n (the index of the image out of total_images)
            # gray code is obtained by xoring the bits of x with itself shifted, and selecting the n-th bit
            return (x^(x>>1))&(1<<(total_images-1-n))!=0
        
        imgs_code = 255*np.fromfunction(xn_to_gray, (total_images, length), dtype=int).astype(np.uint8)
        return imgs_code
    
    def encode(self, proj_wh, flipped_patterns=True):
        """
        encode projector's width and height into gray code patterns
        :param proj_wh: projector's (width, height) in pixels as a tuple
        :param flipped_patterns: if True, flipped patterns are also generated for better binarization
        :return: a 3D numpy array of shape (total_images, height, width) where total_images is the number of gray code patterns
        """
        width, height = proj_wh
        codes_width_1d = self.encode1d(width)[:, None, :]
        codes_width_2d = codes_width_1d.repeat(height, axis=1)
        codes_height_1d = self.encode1d(height)[:, :, None]
        codes_height_2d = codes_height_1d.repeat(width, axis=2)
        
        img_white = np.full((height, width), 255, dtype=np.uint8)[None, ...]
        img_black = np.full((height, width),  0, dtype=np.uint8)[None, ...]
        all_images = np.concatenate((codes_width_2d, codes_height_2d), axis=0)
        if flipped_patterns:
            all_images = np.concatenate((all_images, 255-codes_width_2d), axis=0)
            all_images = np.concatenate((all_images, 255-codes_height_2d), axis=0)
        all_images = np.concatenate((all_images, img_white, img_black), axis=0)
        return all_images[..., None]

gray = GrayCode()
graycode_patterns = gray.encode((PROJ_RES_W, PROJ_RES_H), flipped_patterns=True)
patterns_dir.mkdir(parents=True, exist_ok=True)
for i, pattern in enumerate(graycode_patterns):
    Image.fromarray(np.tile(pattern, (1, 1, 3))).save(patterns_dir / f"graycode_{i:03d}.png")

N_TEXTURES_PER_VIEW = len(graycode_patterns)
TEXTURES = np.array(["graycode_{:03d}".format(i) for i in range(len(graycode_patterns))])


def get_scene_resolution(scene):
    resolution_scale = (scene.render.resolution_percentage / 100.0)
    resolution_x = scene.render.resolution_x * resolution_scale  # [pixels]
    resolution_y = scene.render.resolution_y * resolution_scale  # [pixels]
    return int(resolution_x), int(resolution_y)


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def get_c2w_opencv(camera):
    """
    converts a c2w matrix of blender, into opencv c2w (x is right, y is down, z goes into view direction)
    """
    R_bcam2cv = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    c2w = np.matmul(np.array(camera.matrix_world), R_bcam2cv)
    return c2w
    
    
def get_camera_parameters_intrinsic(scene, camera):
    """ Get intrinsic camera parameters: focal length and principal point. """
    # ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera/120063#120063
    focal_length = camera.data.lens  # [mm]
    res_x, res_y = get_scene_resolution(scene)
    cam_data = camera.data
    sensor_size_in_mm = get_sensor_size(cam_data.sensor_fit, cam_data.sensor_width, cam_data.sensor_height)
    sensor_fit = get_sensor_fit(
        cam_data.sensor_fit,
        scene.render.pixel_aspect_x * res_x,
        scene.render.pixel_aspect_y * res_y
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = res_x
    else:
        view_fac_in_px = pixel_aspect_ratio * res_y
    pixel_size_mm_per_px = (sensor_size_in_mm / focal_length) / view_fac_in_px
    skew = 0
    f_x = 1.0 / pixel_size_mm_per_px
    f_y = (1.0 / pixel_size_mm_per_px) / pixel_aspect_ratio
    c_x = (res_x) / 2.0 - cam_data.shift_x * view_fac_in_px
    c_y = (res_y) / 2.0 + (cam_data.shift_y * view_fac_in_px) / pixel_aspect_ratio
    K = np.array([[f_x, skew, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    return K

def parent_obj_to_camera(object, a_parent):
    object.parent = a_parent  # setup parenting
    return b_empty


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list
    

def generate_all_black(height, width, dst=None):
    img = Image.new("RGB", (width, height), "black")
    if dst is not None:
        img.save(str(dst))
    return np.array(img)

def generate_all_white(height, width, dst=None):
    img = Image.new("RGB", (width, height), "white")
    if dst is not None:
        img.save(str(dst))
    return np.array(img)

def remove_textures():
    # remove current textures
    exceptions = ["checkerboard.png"]
    for image in bpy.data.images:
        if image.name not in exceptions:
            print("removing texture: {}".format(image))
            bpy.data.images.remove(image)

def load_texture(np_image, texture_key, proj_h, proj_w):
    image = np_image
    if image.shape[0] != proj_h or image.shape[1] != proj_w:
        image = cv2.resize(image, (proj_w, proj_h))
    float_texture = (image / 255).astype(np.float32)
    # flipped_texture = np.flip(float_texture, axis=0)
    padded_texture = np.concatenate((float_texture, np.ones_like(float_texture)[:, :, 0:1]), axis=-1)
    bpy_image = bpy.data.images.new(texture_key, width=proj_w, height=proj_h, alpha=False)
    bpy_image.pixels.foreach_set(padded_texture.ravel())
    bpy_image.pack()

def pack_texture(np_texture, texture_key, proj_h, proj_w):
    image = np_texture
    if image.shape[0] != proj_h or image.shape[1] != proj_w:
        image = cv2.resize(image, (proj_w, proj_h))
    float_texture = (image / 255).astype(np.float32)
    # flipped_texture = np.flip(float_texture, axis=0)
    padded_texture = np.concatenate((float_texture, np.ones_like(float_texture)[:, :, 0:1]), axis=-1)
    bpy_image = bpy.data.images.new(texture_key, width=proj_w, height=proj_h, alpha=False)
    bpy_image.pixels.foreach_set(padded_texture.ravel())
    bpy_image.pack()

def save_texture(texture_key, proj_width, proj_height, dst):
    # save current texture
    if dst.is_file():
        return
    print("saving to: {}".format(str(dst)))
    image = np.array(bpy.data.images[texture_key].pixels).reshape(proj_height, proj_width, 4)
    image = Image.fromarray((image[:, :, :3]*255).astype(np.uint8))
    image.save(dst)

def pack_textures(textures, proj_width, proj_height):
    # add new textures
    for key in textures.keys():
        if key not in bpy.data.images.keys():
            # flipped_texture = np.flip(textures[key], axis=0)
            padded_texture = np.concatenate((textures[key], np.ones_like(textures[key])[:, :, 0:1]), axis=-1)
            bpy_image = bpy.data.images.new(key, width=proj_width, height=proj_height, alpha=False)
            bpy_image.pixels.foreach_set(padded_texture.ravel())
            bpy_image.pack()
        
def swap_projector_texture(texture_name):
    projector_name = "Projector"
    bpy.data.images[texture_name].colorspace_settings.name = 'Linear'
    bpy.data.lights[projector_name].node_tree.nodes["Image Texture"].image = bpy.data.images[texture_name]


def hide_object_and_children(obj, hide=True):
    # hide the children
    obj.hide_viewport = hide
    obj.hide_render = hide
    for child in obj.children:
        child.hide_viewport = hide
        child.hide_render = hide

def randomize_checkerboard(obj, initial_position, scene):
    is_in_camera_frustrum = False
    is_in_projector_frustrum = False
    max_attempts = 1000
    obj.select_set(True)
    attempt = 0
    while (((not is_in_camera_frustrum) or (not is_in_projector_frustrum)) and attempt < max_attempts):
        # obj.location = initial_position
        # obj.rotation_euler = (0, 90, 0)
        np.random.seed(random.randint(0, 10000))
        euler = np.random.uniform(low=-np.pi / 6, high=np.pi / 6, size=(3,))
        loc = np.random.uniform(low=-0.1, high=0.3, size=(3,))
        obj.location = tuple(loc + initial_position)
        obj.rotation_euler = tuple(euler + (0, np.pi / 2, 0))
        bpy.context.view_layer.update()
        coords = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
        coords_2d_cam = np.array([world_to_camera_view(scene, scene.objects["Camera"], coord) for coord in coords])
        is_in_camera_frustrum = (np.all(coords_2d_cam >= 0.0, axis=1) & np.all(coords_2d_cam[:, :2] <= 1.0, axis=1)).all()
        coords_2d_proj = np.array([world_to_camera_view(scene, scene.objects["Projector_Camera"], coord) for coord in coords])
        is_in_projector_frustrum = (np.all(coords_2d_proj >= 0.0, axis=1) & np.all(coords_2d_proj[:, :2] <= 1.0, axis=1)).all()
        attempt += 1

####################################################################################
################################ Configure Scene ###################################
####################################################################################
if not patterns_dir.is_dir():
    patterns_dir.mkdir(exist_ok=True, parents=True)

# data to store in JSON file
proj1_intrinsics = get_camera_parameters_intrinsic(bpy.context.scene, bpy.context.scene.objects["Projector_Camera"])
proj1_intrinsics[0, :] /= RESOLUTION
proj1_intrinsics[0, :] *= PROJ_RES_W
proj1_intrinsics[1, :] /= RESOLUTION
proj1_intrinsics[1, :] *= PROJ_RES_H
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    'K_cam': listify_matrix(get_camera_parameters_intrinsic(bpy.context.scene, bpy.context.scene.camera)),
    'K_proj': listify_matrix(proj1_intrinsics),
    'blender_matrix_world_proj': listify_matrix(bpy.context.scene.objects["Projector"].matrix_world),
}
# get textures
remove_textures()
texture_names = np.array(TEXTURES)

bpy.context.scene.objects['Projector'].data.energy = 50
bpy.context.scene.objects["Camera_Light"].data.energy = 10
bpy.context.scene.render.use_persistent_data = True
if SINGLE_BOUNCE:
    bpy.context.scene.cycles.max_bounces = 0
else:
    bpy.context.scene.cycles.max_bounces = 3
if AMBIENT_LIGHT:
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0.01
else:
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0
if COLOC_LIGHT:
    bpy.context.scene.objects["Camera_Light"].hide_render = False
    bpy.context.scene.objects["Camera_Light"].hide_viewport = False
else:
    bpy.context.scene.objects["Camera_Light"].hide_render = True
    bpy.context.scene.objects["Camera_Light"].hide_viewport = True
if N_VIEWS == 1:
    hide_object_and_children(bpy.context.scene.objects["checkerboard"], True)
    hide_object_and_children(bpy.context.scene.objects["projection_target"], False)
else:
    hide_object_and_children(bpy.context.scene.objects["checkerboard"], False)
    hide_object_and_children(bpy.context.scene.objects["projection_target"], True)
# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
# Add passes for additionally dumping albedo and normals.
bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

# Remove all nodes from current compositor
for node in tree.nodes:
    tree.nodes.remove(node)

# Add from scratch nodes in compositor
if 'Custom Outputs' not in tree.nodes:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    if OUTPUT_DEPTH:
        if FORMAT == 'OPEN_EXR':
            links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        else:
            # Remap as other types can not represent the full range of depth.
            map = tree.nodes.new(type="CompositorNodeMapRange")
            # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
            map.inputs['From Min'].default_value = 0
            map.inputs['From Max'].default_value = 8
            map.inputs['To Min'].default_value = 1
            map.inputs['To Max'].default_value = 0
            links.new(render_layers.outputs['Depth'], map.inputs[0])

            links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.name = 'Normal Output'
    if OUTPUT_NORMALS:
        links.new(render_layers.outputs[normal_layer_name], normal_file_output.inputs[0])

    if CONSTANT_BG:
        alpha_over = tree.nodes.new(type="CompositorNodeAlphaOver")
        alpha_over.label = 'Alpha Over'
        alpha_over.name = 'Alpha Over'
        #alpha_over.premul = 1
        alpha_over.inputs[1].default_value = (0, 0, 0, 1)
        links.new(render_layers.outputs[denoised_layer_name], alpha_over.inputs[2])

        image_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        image_file_output.label = 'Image Output'
        image_file_output.name = 'Image Output'
        links.new(alpha_over.outputs[0], image_file_output.inputs[0])
    else:
        image_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        image_file_output.label = 'Image Output'
        image_file_output.name = 'Image Output'
        links.new(render_layers.outputs[denoised_layer_name], image_file_output.inputs[0])
    image_file_output.format.color_mode = 'RGBA'
# Background

bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background

objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})
scene = bpy.context.scene
scene.render.resolution_percentage = 100
cam = scene.objects['Camera']
cam_light = scene.objects["Camera_Light"]
projector = scene.objects['Projector']
projector_camera = scene.objects['Projector_Camera']
checkerboard = scene.objects['checkerboard']
for c in cam.constraints:
    cam.constraints.remove(c)
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
cam.location = CAMERA_LOC
cam_light.location = CAMERA_LOC
projector.location = PROJECTOR_LOC
projector_camera.location = PROJECTOR_LOC
checkerboard.location = CHECKERBOARD_LOC
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
proj_constraint = projector.constraints[0]
proj_cam_constraint = projector_camera.constraints[0]
origin = (0, 0, 0)
b_empty = bpy.data.objects.new("Empty", None)
b_empty.location = origin
parent_obj_to_camera(cam, b_empty)
parent_obj_to_camera(cam_light, b_empty)
#parent_obj_to_camera(light, b_empty)
scene.collection.objects.link(b_empty)
bpy.context.view_layer.objects.active = b_empty
# scene.objects.active = b_empty
cam_constraint.target = b_empty
proj_constraint.target = b_empty
proj_cam_constraint.target = b_empty
scene.render.image_settings.file_format = 'PNG'  # set output format to .png

for output_node in [tree.nodes['Depth Output'], tree.nodes['Normal Output'], tree.nodes['Image Output']]:
    output_node.base_path = ''

out_data['frames'] = []

selected_textures = np.tile(texture_names, N_VIEWS)
# prepare camera locations sequence
if RANDOM_VIEWS:
    cam_locations = []
    for i in range(len(selected_textures) // N_TEXTURES_PER_VIEW):
        rot = np.random.uniform(0, 1, size=3) * (1, 0, 2*np.pi)
        rot[0] = np.arccos(2 * rot[0] - 1) / 2
        #b_empty.rotation_euler = rot
        r = CAM_DIST_FROM_ZERO
        new_loc = (r*np.sin(rot[0])*np.cos(rot[2]), r*np.sin(rot[0])*np.sin(rot[2]), r*np.cos(rot[0]))
        cam_locations.append(new_loc)
    cam_locations = np.stack(cam_locations)
    cam_locations = cam_locations.repeat(N_TEXTURES_PER_VIEW, axis=0)
else:
    cam_locations = [cam.location]*N_TEXTURES_PER_VIEW*N_VIEWS

view_ids = np.arange(len(cam_locations))

print("view_ids: {}".format(view_ids))
print("selected_textures: {}".format(selected_textures))
####################################################################################
############################### Main Render Loop ###################################
####################################################################################
if DEBUG:
    cam.location = cam_locations[DEBUG_VIEW_INDEX]
    cam_light.location = cam_locations[DEBUG_VIEW_INDEX]
    raise NotImplementedError("DEBUG")
for i in range(0, len(selected_textures)):
    cam.location = cam_locations[i]
    cam_light.location = cam_locations[i]
    if N_VIEWS == 1:
        cur_session_dir = results_dir
    else:
        cur_session_dir = Path(results_dir, "session_{:02d}".format(i // N_TEXTURES_PER_VIEW))
        cur_session_dir.mkdir(parents=True, exist_ok=True)
        if i % N_TEXTURES_PER_VIEW == 0:
            randomize_checkerboard(checkerboard, CHECKERBOARD_LOC, scene)
    print("session: {} / {}, pattern: {} / {}".format(i // N_TEXTURES_PER_VIEW, N_VIEWS, i % N_TEXTURES_PER_VIEW, N_TEXTURES_PER_VIEW))
    selected_texture = selected_textures[i]
    cur_image = np.tile(graycode_patterns[i % N_TEXTURES_PER_VIEW], (1, 1, 3))
    if i < N_TEXTURES_PER_VIEW:
        load_texture(cur_image, selected_texture, PROJ_RES_H, PROJ_RES_W)
    swap_projector_texture(selected_texture)
    bpy.context.view_layer.update()
    my_path_str = str(Path(cur_session_dir, "{:04d}".format(i % N_TEXTURES_PER_VIEW)))
    if OUTPUT_DEPTH:
        tree.nodes['Depth Output'].file_slots[0].path = str(my_path_str) + "_depth"
    if OUTPUT_NORMALS:
        tree.nodes['Normal Output'].file_slots[0].path = str(my_path_str) + "_normal"
    tree.nodes['Image Output'].file_slots[0].path = str(my_path_str)
    frame_data = {
        'file_path': Path(my_path_str).stem + ".png",
        'blender_matrix_world': listify_matrix(cam.matrix_world),
        'RT': listify_matrix(get_c2w_opencv(cam)),
        'patterns': [str(selected_texture)]
    }
    out_data['frames'].append(frame_data)
    with open(Path(results_dir, 'transforms.json'), 'w') as out_file:
        json.dump(out_data, out_file, indent=4)
    if not DEBUG:
        bpy.ops.render.render()
        #  blender hacks to change output name.
        outRenderFileNamePadded = Path(my_path_str + "depth{:04d}.png".format(BLENDER_SUFFIX))
        outRenderFileName = Path(my_path_str + "depth.png")
        if outRenderFileName.is_file():
            outRenderFileName.unlink()
        if outRenderFileNamePadded.is_file():
            outRenderFileNamePadded.rename(outRenderFileName)
        outRenderFileNamePadded = Path(my_path_str + "normal{:04d}.png".format(BLENDER_SUFFIX))
        outRenderFileName = Path(my_path_str + "normal.png")
        if outRenderFileName.is_file():
            outRenderFileName.unlink()
        if outRenderFileNamePadded.is_file():
            outRenderFileNamePadded.rename(outRenderFileName)
        outRenderFileNamePadded = Path(my_path_str + "{:04d}.png".format(BLENDER_SUFFIX))
        outRenderFileName = Path(my_path_str + ".png")
        if outRenderFileName.is_file():
            outRenderFileName.unlink()
        if outRenderFileNamePadded.is_file():
            outRenderFileNamePadded.rename(outRenderFileName)
        # dst = Path(patterns_dir, selected_texture + ".png")
        # save_texture(selected_texture, PROJ_RES_W, PROJ_RES_H, dst)
