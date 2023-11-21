# shows some use cases of calibration and reconstruction

import gsoup
from pathlib import Path
import numpy as np

# instantiate a gray code object
gray = gsoup.GrayCode()
proj_wh = (800, 800)
mode = "ij"
orig_patterns = gray.encode(proj_wh)  # not really used but just to see the patterns
patterns = gsoup.load_images(
    Path("tests/tests_resource/correspondence"),
)  # load captured images
cam_wh = (patterns[0].shape[1], patterns[0].shape[0])
forward_map, fg = gray.decode(
    patterns,
    (800, 800),
    output_dir="resource/forward",
    debug=True,
    mode=mode,
)  # decode to recieve a dense map from camera pixels to projector pixels and a foreground mask
# we couldve just set the camera transform to be iden. but we want to later compare with the ground truth in blender
blend_to_cv = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
cam_transform = np.array([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
cam_transform = cam_transform @ blend_to_cv
### gt info. ###
# cam_int = np.array([[800, 0, 400.0], [0.0, 800, 400], [0, 0, 1]])
# proj_int = np.array([[800, 0, 400.0], [0.0, 800, 400], [0, 0, 1]])
# proj_transform = np.array([[-0.12403473, -0.23891242,  0.96308672,  0.8 ],
#                             [ 0.99227786, -0.02986405,  0.12038584,  0.1],
#                             [ 0.        ,  0.97058171,  0.24077168,  0.2],
#                             [ 0.        ,  0.        ,  0.        ,  1. ]])
# proj_transform = proj_transform @ blend_to_cv
# cam_dist = None
### end gt info. ###
### calib ###
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
cam_int, cam_dist, proj_int, proj_dist, proj_transform = (
    result["cam_intrinsics"],
    result["cam_distortion"],
    result["proj_intrinsics"],
    result["proj_distortion"],
    result["proj_transform"],
)
proj_transform = cam_transform @ np.linalg.inv(proj_transform)  # p2w = c2w @ p2c
### end calib ###

# visiualize
# calibration_static_view(cam_transform, proj_transform, (800, 800), (800, 800), cam_int, cam_dist, proj_int, forward_map, fg, mode)
# reconstruct point cloud of scene
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
gsoup.save_pointcloud(pc, "resource/points.ply")
