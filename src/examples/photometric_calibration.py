import gsoup
import numpy as np
from pathlib import Path
import mitsuba as mi


def create_scene(proj_wh, cam_wh):
    """
    helper function creating a mitsuba scene with a projector-camera pair
    """
    projector_scene = gsoup.ProjectorScene()
    proj_cv_K = np.array(
        [
            [proj_wh[0], 0.0, proj_wh[0] / 2],
            [0.0, proj_wh[1], proj_wh[1] / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    cam_cv_K = np.array(
        [
            [cam_wh[0], 0.0, cam_wh[0] / 2],
            [0.0, cam_wh[1], cam_wh[1] / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    projector_scene.create_default_scene(
        proj_wh=proj_wh,
        cam_wh=cam_wh,
        cam_cv_K=cam_cv_K,
        proj_cv_K=proj_cv_K,
        # proj_fov=45.0,
        # cam_fov=45.0,
        proj_brightness=2.0,
        spp=256,
    )
    transform = (
        mi.ScalarTransform4f().look_at(
            origin=[0.0, 0.0, 0.0],  # along +X axis
            target=[-1, 0, 0],
            up=[0, 0, 1],  # Z-up
        ),
    )
    projector_scene.set_projector_transform(np.array(transform[0].matrix))
    projector_scene.set_camera_transform(np.array(transform[0].matrix))
    return projector_scene


def simulate_procam(patterns_to_project, scene, synth_V=None):
    """
    helper function that simulates a video projector projecting some patterns onto a scene
    :param patterns_to_project: a dictionary of numpy images to project
    :param scene: a class containing a mitsuba scene (see procam.py)
    :param synth_V: a synthetic color mixing matrix per-pixel, simulating real artifacts created by the projector
    """
    captures = {}
    ### simulate procam ###
    for i, pattern_name in enumerate(sorted(patterns_to_project.keys())):
        pattern = patterns_to_project[pattern_name]
        scene.set_projector_texture(pattern)
        capture = scene.capture(raw=True)
        if synth_V is not None:
            capture_expanded = capture[..., None]  # (H, W, 3, 1)
            mixed = np.matmul(synth_V, capture_expanded)  # (H, W, 3, 1)
            capture = capture_expanded[..., 0]  # remove singleton dimension → (H, W, 3)
        captures[pattern_name] = capture
    ### end simulate procam ###
    return captures


if __name__ == "__main__":
    print("Photometric Calibration Example")
    proj_wh = (512, 512)
    cam_wh = (512, 512)
    low_val = 80 / 255
    high_val = 170 / 255
    n_samples_per_channel = 20
    ################## offline steps for photometric calibration ##################
    # 0. create a scene
    scene = create_scene(proj_wh, cam_wh)
    # 1. create patterns for calibration
    # test_texture = gsoup.generate_voronoi_diagram(512, 512, 1000)
    # test_texture = gsoup.to_float(test_texture)
    patterns = {
        # "test_image": test_texture,
        "all_black": np.zeros(proj_wh + (3,), dtype=np.float32),
        "off_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "red_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "green_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "blue_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "on_image": np.ones(proj_wh + (3,), dtype=np.float32) * high_val,
        "white_image": np.ones(proj_wh + (3,), dtype=np.float32),
    }
    patterns["red_image"][:, :, 0] = high_val
    patterns["green_image"][:, :, 1] = high_val
    patterns["blue_image"][:, :, 2] = high_val
    input_values = np.linspace(0.0, 1.0, num=20)
    for i in range(n_samples_per_channel):
        patterns["gray_{:03d}".format(i)] = (
            np.ones(proj_wh + (3,), dtype=np.float32) * input_values[i]
        )
    # 2. project patterns and acquire images (also simulate a global color mixing matrix)
    # synth_V = np.array([[0.9, 0.1, 0.1], [0.2, 0.8, 0.2], [0.1, 0.1, 0.9]])
    captured = simulate_procam(patterns, scene, synth_V=None)
    # save the captured images
    for name, img in captured.items():
        gsoup.save_image(img, f"resource/photometric_compensation/{name}.png")
    # 3. we use a "linear" camera here, if not possible we need to find camera response function per channel
    # 4. and linearize camera response function
    # 5. no need to white balance camera channels, color mixing matrix will take care of that
    # 6. find inverse of color mixing matrix per-pixel
    inv_v = gsoup.estimate_color_mixing_matrix(
        off_image=captured["off_image"],
        red_image=captured["red_image"],
        green_image=captured["green_image"],
        blue_image=captured["blue_image"],
        cam_inv_response=None,
    )
    # 7. find projector inverse response function
    measured = np.stack(
        [captured["gray_{:03d}".format(i)] for i in range(n_samples_per_channel)],
        axis=0,
    )
    proj_response = gsoup.estimate_projector_inverse_response(
        measured,
        input_values=input_values,
        fg_mask=None,
    )
    ################## online steps for photometric calibration ##################
    # 1. load/create pattern to project
    # texture = gsoup.generate_voronoi_diagram(512, 512, 1000)
    # texture_float = gsoup.to_float(texture)
    texture_float = np.ones((512, 512, 3), dtype=np.float32) * 0.5
    # 2. compute compensation image
    compensation_image = gsoup.compute_compensation_image(
        texture_float,
        inv_v,
        cam_inverse_response=None,
        proj_inverse_response=proj_response,
    )
    # 3. project compensation image and uncompensated for comaprisons
    result = simulate_procam(
        {"compensation_image": compensation_image, "texture_float": texture_float},
        scene,
    )
    gsoup.save_image(
        result["compensation_image"],
        Path("resource/photometric_compensation/_compensated.png"),
    )
    gsoup.save_image(
        result["texture_float"],
        Path("resource/photometric_compensation/_uncompensated.png"),
    )
    gsoup.save_image(
        texture_float,
        Path("resource/photometric_compensation/_target.png"),
    )
