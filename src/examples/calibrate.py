# shows some use cases of calibration and reconstruction
import gsoup
from pathlib import Path
import numpy as np

# instantiate a gray code object
gray = gsoup.GrayCode()
# set projector resolution
proj_wh = (800, 800)
# set decoding mode (rows first or columns first)
mode = "ij"  # "ij" is rows first, "xy" is columns first
# generate the gray code patterns, this is the ground truth
orig_patterns = gray.encode(proj_wh)
# save the patterns for projection and capturing
gsoup.save_images(orig_patterns, "tests/tests_resource/patterns/")
### start capture time ###
# here is the point you need to project these patterns onto the scene and capture with a camera
# the captured images should be saved in a folder with increasing numbers or increasing alphanumerical as names
### end capture time ###
# load simulated captured images
captured_patterns = gsoup.load_images(
    Path("tests/tests_resource/correspondence"),
)
# find camera resolution
cam_wh = (
    captured_patterns[0].shape[1],
    captured_patterns[0].shape[0],
)
# finally, decode the captured images
# the result is a dense map from camera pixels to projector pixels and a foreground mask
forward_map, fg = gray.decode(
    captured_patterns,
    (800, 800),
    output_dir="resource/forward",
    debug=True,
    mode=mode,
)
# the map is nice for many tasks, but we want to reconsutrct the scene in 3D, so we need to calibrate the procam pair.
# first, lets decide where the camera is. Usually you will set this as identity (camera is at the origin).
# but this scene was simulated in blender with a different known camera transform, so we will use that transform here.
blend_to_cv = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
cam_transform = np.array([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
cam_transform = cam_transform @ blend_to_cv
### gt information: ###
# cam_int = np.array([[800, 0, 400.0], [0.0, 800, 400], [0, 0, 1]])
# proj_int = np.array([[800, 0, 400.0], [0.0, 800, 400], [0, 0, 1]])
# proj_transform = np.array([[-0.12403473, -0.23891242,  0.96308672,  0.8 ],
#                             [ 0.99227786, -0.02986405,  0.12038584,  0.1],
#                             [ 0.        ,  0.97058171,  0.24077168,  0.2],
#                             [ 0.        ,  0.        ,  0.        ,  1. ]])
# proj_transform = proj_transform @ blend_to_cv
# cam_dist = None
### end gt info. ###
# let's calibrate the procam pair. this requires capturing a chessboard pattern with the camera and projecting the patterns with the projector.
# see the folder for examples of the captured images.
result = gsoup.calibrate_procam(
    (800, 800),
    Path("tests/tests_resource/calibration"),
    chess_vert=15,
    chess_hori=15,
    chess_block_size=0.0185,
    output_dir="resource/calibration",
    projector_orientation="none",
    debug=True,
)
# the result is a dictionary with everything we need to reconstruct the scene
cam_int, cam_dist, proj_int, proj_dist, proj_transform = (
    result["cam_intrinsics"],
    result["cam_distortion"],
    result["proj_intrinsics"],
    result["proj_distortion"],
    result["proj_transform"],
)
# lets find the projector to world transform (the result above is the projector to camera transform)
proj_transform = cam_transform @ np.linalg.inv(proj_transform)  # p2w = c2w @ p2c

# we can visiualize the result using the viewer (commented out)
# calibration_static_view(cam_transform, proj_transform, (800, 800), (800, 800), cam_int, cam_dist, proj_int, forward_map, fg, mode)

# but more importantly, let us reconstruct a point cloud of scene
pc = gsoup.reconstruct_pointcloud(
    forward_map,
    fg,
    cam_transform,
    proj_transform,
    cam_int,
    cam_dist,
    proj_int,
    mode=mode,
)
# save the point cloud in a .ply format. this can be viewed in meshlab or other 3D viewers
gsoup.save_pointcloud(pc, "resource/points.ply")
