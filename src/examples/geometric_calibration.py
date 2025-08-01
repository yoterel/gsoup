# shows some use cases of calibration and reconstruction
import gsoup
from pathlib import Path
import numpy as np
import mitsuba as mi


def simulate_procam_correspondence(orig_patterns, proj_wh, cam_wh):
    projector_scene = gsoup.ProjectorScene()
    projector_scene.create_default_scene(
        mesh_file=Path("./tests/tests_resource/gt_mesh.obj"),
        mesh_scale=1.0,
        spp=512,
        proj_wh=proj_wh,
        cam_wh=cam_wh,
        proj_fov=45.0,
        cam_fov=45.0,
    )
    transform = (
        mi.ScalarTransform4f().look_at(
            origin=[1.0, 0.0, 0.0],  # along +X axis
            target=[0, 0, 0],
            up=[0, 0, 1],  # Z-up
        ),
    )
    projector_scene.set_projector_transform(np.array(transform[0].matrix))
    projector_scene.set_camera_transform(np.array(transform[0].matrix))
    captures = []
    ### simulate procam ###
    for i, pattern in enumerate(orig_patterns):
        projector_scene.set_projector_texture(pattern)
        capture = projector_scene.capture()
        captures.append(capture)
    ### end simulate procam ###
    return np.array(captures)


def simulate_procam_calibration(orig_patterns, proj_wh, cam_wh):
    projector_scene = gsoup.ProjectorScene()
    projector_scene.create_default_scene(
        # mesh_file=Path("./tests/tests_resource/gt_mesh.obj"),
        # mesh_scale=1.0,
        spp=512,
        proj_wh=proj_wh,
        cam_wh=cam_wh,
        proj_fov=45.0,
        cam_fov=45.0,
    )
    transform = (
        mi.ScalarTransform4f().look_at(
            origin=[1.0, 0.0, 0.0],  # along +X axis
            target=[0, 0, 0],
            up=[0, 0, 1],  # Z-up
        ),
    )
    projector_scene.set_projector_transform(np.array(transform[0].matrix))
    projector_scene.set_camera_transform(np.array(transform[0].matrix))
    projector_scene.transform_square_randomly()
    captures = []
    ### simulate procam ###
    for i, pattern in enumerate(orig_patterns):
        projector_scene.set_projector_texture(pattern)
        capture = projector_scene.capture()
        captures.append(capture)
    ### end simulate procam ###
    return np.array(captures)


if __name__ == "__main__":
    print("Geometric Calibration + Reconstruction Example")
    # instantiate a gray code object
    gray = gsoup.GrayCode()
    # set projector resolution
    proj_wh = (400, 400)
    # set decoding mode (rows first or columns first)
    mode = "ij"  # "ij" is rows first, "xy" is columns first
    # generate the gray code patterns for structured light projection
    orig_patterns = gray.encode(proj_wh)[0:1]
    ###################
    ### let's calibrate the procam pair. this requires capturing a checkerboard with the camera and projecting structured light patterns with the projector.
    n_sessions = 10
    # simulate calibration sessions
    for i in range(n_sessions):
        captured_patterns_float = simulate_procam_calibration(
            orig_patterns, proj_wh, proj_wh
        )
        captured_patterns = gsoup.to_8b(captured_patterns_float)
        gsoup.save_images(
            captured_patterns,
            "resource/geometric_calibration/calibration_captured/session_{:02d}".format(
                i
            ),
        )
    breakpoint()
    # actual calibration
    result = gsoup.calibrate_procam(
        proj_wh,
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
    ###################
    ### let's establish correspondence between projector and camera pixels.
    # simulate the same procam pair we calibrated, this time with a mesh to project onto.
    captured_patterns_float = simulate_procam_correspondence(
        orig_patterns, proj_wh, proj_wh
    )
    captured_patterns = gsoup.to_8b(captured_patterns_float)
    # save them in case we want to inspect later
    gsoup.save_images(
        captured_patterns,
        "resource/geometric_calibration/correspondence_captured",
    )
    # commented out: here we used blender to do this.
    # load simulated captured images
    # captured_patterns = gsoup.load_images(
    #     Path("tests/tests_resource/correspondence_blender"),
    # )
    # get camera resolution
    cam_wh = (
        captured_patterns[0].shape[1],
        captured_patterns[0].shape[0],
    )
    # finally, decode the captured images
    # TODO: seperate direct and indirect light, and use only direct channel for better inlier detection
    # the result is a dense map from camera pixels to projector pixels and a foreground mask
    forward_map, fg = gray.decode(
        captured_patterns,
        proj_wh,
        output_dir="resource/geometric_calibration/correspondence_forward",
        debug=True,
        mode=mode,
    )
    ###################
    ### now we can reconstruct the scene in 3D using triangulation.
    # commented out: using blender.
    # first, we need the camera extrinsics. Usually you will set this as identity (camera is at the origin).
    # but this scene was simulated in blender with a different known camera transform, so we will use that transform here.
    # blend_to_cv = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # cam_transform = np.array([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    # cam_transform = cam_transform @ blend_to_cv
    ### gt blender information: ###
    # cam_int = np.array([[800, 0, 400.0], [0.0, 800, 400], [0, 0, 1]])
    # proj_int = np.array([[800, 0, 400.0], [0.0, 800, 400], [0, 0, 1]])
    # proj_transform = np.array([[-0.12403473, -0.23891242,  0.96308672,  0.8 ],
    #                             [ 0.99227786, -0.02986405,  0.12038584,  0.1],
    #                             [ 0.        ,  0.97058171,  0.24077168,  0.2],
    #                             [ 0.        ,  0.        ,  0.        ,  1. ]])
    # proj_transform = proj_transform @ blend_to_cv
    # cam_dist = None
    ### end gt blenderinfo. ###
    # reconstruct a point cloud of scene
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
    # save the point cloud. this can be viewed in meshlab or other 3D viewers
    gsoup.save_pointcloud(pc, "resource/geometric_calibration/points.ply")
