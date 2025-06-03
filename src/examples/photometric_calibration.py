import gsoup
import numpy as np
from pathlib import Path


def simulate_procam(patterns_to_project):
    projector_scene = gsoup.ProjectorScene()
    projector_scene.create_default_scene()
    captures = {}
    ### simulate procam ###
    for i, pattern_name in enumerate(sorted(patterns_to_project.keys())):
        pattern = patterns_to_project[pattern_name]
        projector_scene.set_projector_texture(pattern)
        capture = projector_scene.capture(raw=True)
        captures[pattern_name] = capture
    ### end simulate procam ###
    return captures


if __name__ == "__main__":
    print("Photometric Calibration Example")
    proj_wh = (800, 600)
    low_val = 80 / 255
    high_val = 170 / 255
    n_samples_per_channel = 20
    ################## offline steps for photometric calibration ##################
    # 1. create patterns
    patterns = {
        "off_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "red_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "green_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
        "blue_image": np.ones(proj_wh + (3,), dtype=np.float32) * low_val,
    }
    patterns["r_image"][:, :, 0] = high_val
    patterns["g_image"][:, :, 1] = high_val
    patterns["b_image"][:, :, 2] = high_val
    for i in range(n_samples_per_channel):
        patterns["gray_{:03d}".format(i)] = (
            np.ones(proj_wh + (3,), dtype=np.float32) * i / 20
        )
    # 2. project patterns and acquire images
    captured = simulate_procam(patterns)
    # 3. we use a "linear" camera here, if not possible we need to find camera response function per channel
    # 4. and linearize camera response function
    # 5. no need to white balance camera channels, color mixing matrix will take care of that
    # 6. find color mixing matrix per-pixel
    V = gsoup.estimate_color_mixing_matrix(
        captured["off_image"],
        captured[]"red_image"],
        captured["green_image"],
        captured["blue_image"],
        cam_inv_response=None,
        bump_value_low=int(low_val*255),
        bump_value_high=int(high_val*255),
    )
    # 7. find projector inverse response function
    measured = np.stack(
        [captured["gray_{:03d}".format(i)] for i in range(n_samples_per_channel)],
        axis=0,
    )
    input_values = np.arange(n_samples_per_channel) / 20
    proj_response = gsoup.estimate_projector_inverse_response(
        measured,
        captured["gray_019"],
        cam_inv_response=None,
        bump_value_low=int(low_val * 255),
        bump_value_high=int(high_val * 255),
    )
    ################## online steps for photometric calibration ##################
    # 1. load/create pattern to project
    texture = gsoup.generate_voronoi_diagram(512, 512, 1000)
    projected_texture = gsoup.to_float(texture)
    # 2. compute compensation image
    compensation_image = gsoup.compute_compensation_image(
        orig_image,
        Vinv,
        cam_inverse_response=None,
        proj_inverse_response=None,
    )
    # 3. project compensation image
    result = simulate_procam(
        {"compensation_image": compensation_image}
    )
    gsoup.save_image(result["compensation_image"], "compensation_image.png")