import pytest
import gsoup
import numpy as np
from pathlib import Path


def test_synthetic_projector():
    texture = gsoup.generate_voronoi_diagram(512, 512, 1000)
    projected_texture = gsoup.to_float(texture)
    projector_scene = gsoup.ProjectorScene()
    projector_scene.create_default_scene(proj_texture=projected_texture)
    # projector_scene.set_projector_transform(
    #     np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # )  # this is the projector to world transform, it is identity in this case
    # projector_scene.set_camera_transform(
    #     np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # )  # this is the projector to world transform, it is identity in this case
    render = projector_scene.render()
    gsoup.save_image(render, "resource/synth_projector.png")
    texture2 = gsoup.generate_voronoi_diagram(200, 600, 1000)
    projector_scene.set_projector_texture(texture2)
    render = projector_scene.render()
    gsoup.save_image(render, "resource/synth_projector2.png")


def test_procam():
    gray = gsoup.GrayCode()
    patterns = gray.encode((128, 128))
    mode = "ij"
    forward_map, fg = gray.decode(
        patterns, (128, 128), output_dir=Path("resource/pix2pix"), mode=mode, debug=True
    )
    backward_map = gsoup.compute_backward_map(
        (128, 128), forward_map, fg, output_dir=Path("resource/pix2pix"), debug=True
    )
    desired = gsoup.generate_lollipop_pattern(128, 128)
    warp_image = gsoup.warp_image(
        backward_map,
        desired,
        cam_wh=(forward_map.shape[1], forward_map.shape[0]),
        mode=mode,
        output_path=Path("resource/warp.png"),
    )
    assert warp_image.shape == (128, 128, 3)
    assert warp_image.dtype == np.uint8
    assert (
        np.mean(np.abs(desired - warp_image)) < 10
    )  # identity correspondence & warp should be very similar
    # calibration_dir = Path("resource/calibration")
    # calibration_dir.mkdir(exist_ok=True, parents=True)
    checkerboard = gsoup.generate_checkerboard(128, 128, 16)
    # T = gsoup.random_perspective()
    # T_opencv = T[:2, :]
    # img_transformed = cv2.warpPerspective(checkerboard, T, (128, 128))
    # captures = np.bitwise_and(patterns==255, checkerboard[None, ...]==1.0)
    # gsoup.save_images(captures, Path(calibration_dir, "0"))
    # gsoup.save_images(captures, Path(calibration_dir, "1"))
    gsoup.save_image(checkerboard, Path("resource/checkerboard.png"))
    #############
    # patterns = gray.encode((800, 800))
    patterns = gsoup.load_images(Path("tests/tests_resource/correspondence_blender"))
    cam_wh = (patterns[0].shape[1], patterns[0].shape[0])
    proj_wh = (800, 800)
    forward_map, fg = gray.decode(
        patterns, (800, 800), output_dir="resource/forward", debug=True, mode=mode
    )
    backward_map = gsoup.compute_backward_map(
        (800, 800),
        forward_map,
        fg,
        mode=mode,
        output_dir="resource/backward_not_interp",
        debug=True,
        interpolate=False,
    )
    backward_map = gsoup.compute_backward_map(
        (800, 800),
        forward_map,
        fg,
        mode=mode,
        output_dir="resource/backward_interp",
        debug=True,
        interpolate=True,
    )
    desired = gsoup.generate_lollipop_pattern(800, 800)
    warp_image = gsoup.warp_image(
        backward_map,
        desired,
        cam_wh=cam_wh,
        mode=mode,
        output_path=Path("resource/debug/warp.png"),
    )
    blend_to_cv = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cam_transform = np.array([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    cam_transform = cam_transform @ blend_to_cv
    ### gt ###
    # cam_int = np.array([[800, 0, 400.0], [0.0, 800, 400], [0, 0, 1]])
    # proj_int = np.array([[800, 0, 400.0], [0.0, 800, 400], [0, 0, 1]])
    # proj_transform = np.array([[-0.12403473, -0.23891242,  0.96308672,  0.8 ],
    #                             [ 0.99227786, -0.02986405,  0.12038584,  0.1],
    #                             [ 0.        ,  0.97058171,  0.24077168,  0.2],
    #                             [ 0.        ,  0.        ,  0.        ,  1. ]])
    # proj_transform = proj_transform @ blend_to_cv
    # cam_dist = None
    ### end gt ###
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
    # calibration_static_view(cam_transform, proj_transform, (800, 800), (800, 800), cam_int, cam_dist, proj_int, forward_map, fg, mode)
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
