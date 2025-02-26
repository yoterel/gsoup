import pytest
import numpy as np
import torch
import gsoup
from pathlib import Path
from gsoup.viewer_drivers import calibration_static_view


def test_type_conversions():
    test_numpy_bool = np.array([1, 0, 0, 1], dtype=bool)
    test_torch_bool = gsoup.to_torch(test_numpy_bool)

    test_float_from_bool = gsoup.to_float(test_numpy_bool)
    assert test_float_from_bool.dtype == np.float32
    test_8bit_from_bool = gsoup.to_8b(test_numpy_bool)
    assert test_8bit_from_bool.dtype == np.uint8
    test_8bit_from_float = gsoup.to_8b(test_float_from_bool)
    assert test_8bit_from_float.dtype == np.uint8
    test_float_from_8bit = gsoup.to_float(test_8bit_from_float)
    assert test_float_from_8bit.dtype == np.float32
    test_float_from_bool = gsoup.to_float(test_torch_bool)
    assert test_float_from_bool.dtype == torch.float32
    test_8bit_from_bool = gsoup.to_8b(test_torch_bool)
    assert test_8bit_from_bool.dtype == torch.uint8
    test_8bit_from_float = gsoup.to_8b(test_float_from_bool)
    assert test_8bit_from_float.dtype == torch.uint8
    test_float_from_8bit = gsoup.to_float(test_8bit_from_float)
    assert test_float_from_8bit.dtype == torch.float32


def test_broadcast_batch():
    R = np.random.randn(1, 3, 3)
    t = np.random.randn(1, 3)
    R, t = gsoup.broadcast_batch(R, t)
    assert R.shape == (1, 3, 3)
    assert t.shape == (1, 3)
    R = np.random.randn(2, 3, 3)
    t = np.random.randn(2, 3)
    R, t = gsoup.broadcast_batch(R, t)
    assert R.shape == (2, 3, 3)
    assert t.shape == (2, 3)
    R = np.random.randn(2, 3, 3)
    t = np.random.randn(1, 3)
    R, t = gsoup.broadcast_batch(R, t)
    assert R.shape == (2, 3, 3)
    assert t.shape == (2, 3)
    R = np.random.randn(1, 3, 3)
    t = np.random.randn(2, 3)
    R, t = gsoup.broadcast_batch(R, t)
    assert R.shape == (2, 3, 3)
    assert t.shape == (2, 3)
    R = np.random.randn(3)
    t = np.random.randn(3)
    R, t = gsoup.broadcast_batch(R, t)
    assert R.shape == (1, 3)
    assert t.shape == (1, 3)
    R = np.random.randn(1)
    t = np.random.randn(1)
    R, t = gsoup.broadcast_batch(R, t)
    assert R.shape == (1, 1)
    assert t.shape == (1, 1)
    R = np.random.randn(3, 3)
    t = np.random.randn(3)
    with pytest.raises(ValueError):
        R, t = gsoup.broadcast_batch(R, t)


def test_transforms():
    R = np.random.randn(2, 3, 3)
    t = np.random.randn(1, 3)
    Rt = gsoup.compose_rt(R, t)
    assert Rt.shape == (2, 3, 4)

    eye = np.array([1, 0, 0], dtype=np.float32)
    at = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    transform_np = gsoup.look_at_np(eye, at, up)
    transform_torch = gsoup.look_at_torch(
        gsoup.to_torch(eye), gsoup.to_torch(at), gsoup.to_torch(up)
    )
    assert np.allclose(transform_np, gsoup.to_numpy(transform_torch))
    transform_np_opengl = gsoup.look_at_np(eye, at, up, opengl=True)
    transform_torch_opengl = gsoup.look_at_torch(
        gsoup.to_torch(eye), gsoup.to_torch(at), gsoup.to_torch(up), opengl=True
    )
    assert np.allclose(transform_np_opengl, gsoup.to_numpy(transform_torch_opengl))
    normal = torch.tensor([0, 0, 1.0])

    location3d = np.array([0.1, 0.1, 0.1])
    location3d_noise = location3d + np.random.randn(3) * 0.01
    location4d = gsoup.to_hom(location3d)
    location4d_noise = gsoup.to_hom(location3d_noise)
    # sanity on opengl projection
    v2w = gsoup.look_at_np(
        np.array([1.0, 0, 0]), location3d, np.array([0, 0, 1.0]), opengl=True
    )[0]
    w2v = np.linalg.inv(v2w)
    v2c = gsoup.perspective_projection()
    opengl_location = v2c @ w2v @ location4d
    opengl_location = gsoup.homogenize(opengl_location)
    assert np.allclose(opengl_location[:2], np.zeros(2))
    # sanity on opencv projection
    c2w = gsoup.look_at_np(np.array([1.0, 0, 0]), location3d, np.array([0, 0, 1.0]))[0]
    w2c = np.linalg.inv(c2w)
    K = gsoup.opencv_intrinsics_from_opengl_project(v2c, 1, 1)
    opencv_location = K @ gsoup.to_34(w2c) @ location4d
    opencv_location = gsoup.homogenize(opencv_location)
    assert np.allclose(opencv_location[:2], np.ones(2) * 0.5)
    # test opencv / opengl conversion
    opengl_location = v2c @ w2v @ location4d_noise
    opengl_location = gsoup.homogenize(opengl_location)
    opencv_location = K @ gsoup.to_34(w2c) @ location4d_noise
    opencv_location = gsoup.homogenize(opencv_location)
    # x is in same direction, but opengl screen is -1 to 1 (while opencv is 0 to 1)
    assert np.allclose((opencv_location[0] - opengl_location[0] / 2), 0.5)
    # y is in opposite direction, but opengl screen is -1 to 1 (while opencv is 0 to 1)
    assert np.allclose((opencv_location[1] + opengl_location[1] / 2), 0.5)


def test_rotations():
    qvecs = gsoup.random_qvec(10)
    torch_qvecs = torch.tensor(qvecs)
    rotmats = gsoup.qvec2mat(qvecs)
    assert rotmats.shape == (10, 3, 3)
    rotmats = gsoup.qvec2mat(torch_qvecs)
    assert rotmats.shape == (10, 3, 3)
    new_qvecs = gsoup.mat2qvec(rotmats)
    mask1 = torch.abs(new_qvecs - torch_qvecs) < 1e-6
    mask2 = torch.abs(new_qvecs + torch_qvecs) < 1e-6
    assert torch.all(mask1 | mask2)
    rotmat = gsoup.qvec2mat(torch_qvecs[0:1])
    assert rotmat.shape == (1, 3, 3)
    new_qvec = gsoup.mat2qvec(rotmat)
    mask1 = torch.abs(new_qvec[0] - torch_qvecs[0]) < 1e-6
    mask2 = torch.abs(new_qvec[0] + torch_qvecs[0]) < 1e-6
    assert torch.all(mask1 | mask2)
    normal = torch.tensor([0, 0, 1.0])
    random_vectors = gsoup.random_vectors_on_sphere(10, normal=normal)
    assert random_vectors.shape == (10, 3)
    assert (random_vectors @ normal).all() > 0
    normal = torch.tensor([[0, 0, 1.0]]).repeat(10, 1)
    random_vectors = gsoup.random_vectors_on_sphere(10, normal=normal)
    assert random_vectors.shape == (10, 3)
    assert (random_vectors[:, None, :] @ normal[:, :, None]).all() > 0
    rotx = gsoup.rotx(np.pi / 2, degrees=False)
    assert np.allclose(
        rotx, np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    )


def test_homogenize():
    x = np.random.rand(100, 2)
    hom_x = gsoup.to_hom(x)
    assert hom_x.shape == (100, 3)
    hom_x = gsoup.to_hom(hom_x)
    assert hom_x.shape == (100, 4)
    dehom_x = gsoup.homogenize(hom_x)
    assert dehom_x.shape == (100, 3)
    dehom_x = gsoup.homogenize(dehom_x, keepdim=True)
    assert dehom_x.shape == (100, 3)
    x = np.random.rand(3)
    hom_x = gsoup.to_hom(x)
    assert hom_x.shape == (4,)
    dehom_x = gsoup.homogenize(hom_x)
    assert dehom_x.shape == (3,)


def test_normalize_vertices():
    v = np.random.rand(100, 3) * 100
    v_normalized = gsoup.normalize_vertices(v)
    assert (v_normalized < 1.0).all()

    v = torch.rand(100, 3) * 100
    v_normalized = gsoup.normalize_vertices(v)
    assert (v_normalized < 1.0).all()


def test_point_to_line_distance():
    p = np.array([0, 0, 0])
    v0 = np.array([1, 0, 0])
    v1 = np.array([2, 0, 0])
    dist = gsoup.point_line_distance(p, v0, v1)
    assert dist == 0.0
    p = np.array([0, 1, 0])
    dist = gsoup.point_line_distance(p, v0, v1)
    assert dist == 1.0
    v0 = np.array([0, 1, 0])
    v1 = np.array([0, 2, 0])
    p = np.array([0, 0, 0])
    dist = gsoup.point_line_distance(p, v0, v1)
    assert dist == 0.0
    p = np.array([1, 0, 0])
    dist = gsoup.point_line_distance(p, v0, v1)
    assert dist == 1.0
    p = np.array([[0, 0, 1], [1, 0, 0]])
    dist = gsoup.point_line_distance(p, v0, v1)
    assert (dist == np.array([1.0, 1.0])).all()


def test_structures():
    v, f = gsoup.structures.cube()
    assert v.shape[0] == 8
    assert f.shape[0] == 12
    v, f = gsoup.structures.icosehedron()
    assert v.shape[0] == 12
    assert f.shape[0] == 20


def test_parsers():
    # obj save and load
    v, f = gsoup.structures.quad_cube()
    gsoup.save_mesh(v, f, "resource/quad_cube.obj")
    gsoup.save_mesh(v, f, "resource/quad_cube.ply")
    v, f = gsoup.structures.cube()
    gsoup.save_mesh(v, f, "resource/cube.obj")
    v1, f1 = gsoup.load_mesh("resource/cube.obj")
    assert np.allclose(v, v1)
    assert np.allclose(f, f1)
    v, f = gsoup.structures.icosehedron()
    vc = np.random.randint(0, 255, size=v.shape).astype(np.uint8)
    gsoup.save_mesh(v, f, "resource/ico.obj")
    v1, f1 = gsoup.load_mesh("resource/ico.obj")
    assert np.allclose(v, v1)
    assert np.allclose(f, f1)
    # ply save
    gsoup.save_mesh(v, f, "resource/ico.ply")
    gsoup.save_mesh(v, f, "resource/ico_vcolor.ply", vertex_colors=vc)
    gsoup.save_mesh(
        v,
        f,
        "resource/ico_fcolor.ply",
        face_colors=np.random.randint(0, 255, size=f.shape).astype(np.uint8),
    )
    gsoup.save_pointcloud(v, "resource/ico_pc.ply")
    # test ascii ply loader
    v1, f1, vc1 = gsoup.load_mesh("resource/ico_vcolor.ply", return_vert_color=True)
    assert np.allclose(v, v1)
    assert np.allclose(f, f1)
    assert np.allclose(vc, vc1)
    # more use cases
    v1, f1 = gsoup.load_mesh("resource/ico_pc.ply")  # loads a pointcloud as a mesh
    assert np.allclose(v, v1)
    assert f1 is None
    v1 = gsoup.load_pointcloud("resource/ico_pc.ply")
    assert np.allclose(v, v1)
    # test binary ply loader``
    v, channels = gsoup.load_pointcloud(
        "tests/tests_resource/splat.ply", return_vert_norms=True
    )
    assert channels.shape == (v.shape[0], 59)


def test_exr():
    normals = gsoup.read_exr("tests/tests_resource/normal0001.exr")
    gsoup.write_exr(normals, "resource/normals.exr")


def test_image():
    random_mask = gsoup.generate_random_block_mask(4, 2, 3)
    assert random_mask.shape == (3, 4, 4)
    random_mask = gsoup.generate_random_block_mask(4, 2, 1)
    assert random_mask.shape == (4, 4)
    checkboard = gsoup.generate_checkerboard(512, 512, 8)  # H, W, 1
    gsoup.save_image(checkboard, "resource/checkboard.png")
    gsoup.save_images([checkboard], "resource", file_names=["resource/checkboard.png"])
    checkboard_RGB = np.tile(checkboard, (1, 1, 3))  # H, W, 3
    checkboard_RGBA = gsoup.add_alpha(checkboard_RGB, checkboard)
    checkboard_RGBA2 = np.tile(checkboard, (1, 1, 4))  # H, W, 4
    assert (checkboard_RGBA == checkboard_RGBA2).all()
    checkboard_RGB_batch = np.tile(checkboard[None, ...], (10, 1, 1, 3))
    checkboard_RGBA_batch = gsoup.add_alpha(checkboard_RGB_batch, checkboard[None, ...])
    checkboard_RGB = gsoup.alpha_compose(checkboard_RGBA, ~checkboard_RGBA[..., :3])
    assert (checkboard_RGB == 1.0).all()
    checkboard_RGB = gsoup.alpha_compose(
        checkboard_RGBA, bg_color=np.array([0.0, 0.0, 1.0])
    )
    assert (checkboard_RGB[..., -1] == 1.0).all()
    lollipop_path = Path("resource/lollipop.png")
    lollipop = gsoup.generate_lollipop_pattern(512, 512, dst=lollipop_path)
    gsoup.save_image(lollipop, lollipop_path)
    lollipop2 = gsoup.load_image(lollipop_path)
    assert np.allclose(lollipop, lollipop2)
    gsoup.save_images(
        lollipop[None, ...], lollipop_path.parent, file_names=["test_save.png"]
    )
    lollipop_pad = gsoup.pad_to_res(lollipop[None, ...], 512, 1024)
    assert lollipop_pad.shape == (1, 512, 1024, 3)
    lollipop_padded_square = gsoup.pad_to_square(lollipop_pad)
    assert lollipop_padded_square.shape == (1, 1024, 1024, 3)
    lollipop_cropped_square = gsoup.crop_to_square(lollipop_pad)
    assert lollipop_cropped_square.shape == (1, 512, 512, 3)
    lollipop_srgb = gsoup.linear_to_srgb(lollipop)
    lollipop_linear = gsoup.srgb_to_linear(lollipop_srgb)
    assert np.allclose(lollipop, lollipop_linear)
    gsoup.generate_concentric_circles(256, 512, dst=Path("resource/circles.png"))
    gsoup.generate_stripe_pattern(
        256, 512, direction="both", dst=Path("resource/stripe.png")
    )
    gsoup.generate_dot_pattern(512, 256, dst=Path("resource/dots.png"))
    gray1 = gsoup.generate_gray_gradient(256, 256, grayscale=True)
    assert gray1.shape == (256, 256)
    assert len(np.unique(gray1)) == 10
    assert gray1.max() == 255
    gray2 = gsoup.generate_gray_gradient(50, 800, vertical=False)
    assert gray2.shape == (50, 800, 3)
    assert gray2.max() == 255
    gray3 = gsoup.generate_gray_gradient(256, 256, bins=-65)
    assert gray3.max() == 0
    gray4 = gsoup.generate_gray_gradient(256, 256, bins=300)
    assert gray4.max() == 255
    gray5 = gsoup.generate_gray_gradient(1080, 1920, bins=300)
    assert gray5.shape == (1080, 1920, 3)
    assert gray5.max() == 255
    dst = Path("resource/voronoi.png")
    gsoup.generate_voronoi_diagram(512, 512, 1000, dst=dst)
    img = gsoup.load_image(dst)
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.uint8
    img = gsoup.load_image(dst, as_grayscale=True)
    assert img.shape == (512, 512, 1)
    assert img.dtype == np.uint8
    img = gsoup.load_image(dst, as_float=True)
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.float32
    assert (img >= 0.0).all()
    assert (img <= 1.0).all()
    img = gsoup.load_image(dst, channels_last=False)
    assert img.shape == (3, 512, 512)
    img = gsoup.load_image(dst, as_float=True, as_grayscale=True)
    assert img.shape == (512, 512, 1)
    assert img.dtype == np.float32
    assert (img >= 0.0).all()
    assert (img <= 1.0).all()
    img = gsoup.load_image(dst, channels_last=False, as_grayscale=True)
    assert img.shape == (1, 512, 512)
    img = gsoup.load_images([dst])
    assert img.shape == (1, 512, 512, 3)
    img = gsoup.load_images([dst], as_grayscale=True)
    assert img.shape == (1, 512, 512, 1)
    img = gsoup.load_images([dst])
    assert img.shape == (1, 512, 512, 3)
    img = gsoup.load_images([dst, dst, dst, dst])
    assert img.shape == (4, 512, 512, 3)
    resized_img = gsoup.resize_images_naive(img, 256, 256, mode="mean")
    assert resized_img.shape == (4, 256, 256, 3)
    resized_img_float = gsoup.resize_images_naive(
        gsoup.to_float(img), 256, 256, mode="mean"
    )
    assert resized_img_float.shape == (4, 256, 256, 3)
    assert resized_img_float.dtype == np.float32
    resized_img_gray = gsoup.resize_images_naive(img[..., 0:1], 256, 256, mode="mean")
    assert resized_img_gray.shape == (4, 256, 256, 1)
    grid = gsoup.image_grid(resized_img, 2, 2)
    assert grid.shape == (512, 512, 3)
    white_images = gsoup.to_8b(np.ones((4, 256, 256, 3), dtype=np.float32))
    pad = 5
    grid2 = gsoup.image_grid(
        white_images, 2, 2, pad=5, pad_color=np.array([255, 255, 0])
    )
    assert grid2.shape == (512 + pad * 4, 512 + pad * 4, 3)
    img = gsoup.load_images([dst, dst, dst, dst], as_grayscale=True)
    assert img.shape == (4, 512, 512, 1)
    img = gsoup.load_images(
        [dst, dst, dst, dst], as_grayscale=True, channels_last=False
    )
    assert img.shape == (4, 1, 512, 512)
    img = gsoup.load_images([dst, dst, dst, dst], resize_wh=(128, 128))
    assert img.shape == (4, 128, 128, 3)
    img, paths = gsoup.load_images(
        [dst, dst, dst, dst],
        resize_wh=(128, 256),
        as_grayscale=True,
        channels_last=False,
        return_paths=True,
        as_float=True,
        to_torch=True,
    )
    assert len(paths) == 4
    assert img.dtype == torch.float32
    assert img.shape == (4, 1, 256, 128)
    lollipop_path1 = Path("resource/lp_ref.png")
    lollipop_path2 = Path("resource/lp_01.png")
    lollipop_path3 = Path("resource/lp_02.png")
    img_ref = gsoup.generate_lollipop_pattern(512, 512, dst=lollipop_path1)
    img1 = gsoup.to_8b(gsoup.to_float(img_ref) / 2)
    gsoup.save_image(img1, lollipop_path2)
    img2 = gsoup.generate_lollipop_pattern(512, 512, dst=lollipop_path3)
    dist1 = gsoup.compute_color_distance(img_ref, img1)
    dist2 = gsoup.compute_color_distance(img_ref, img2)
    assert dist1 < dist2
    test = np.zeros((1, 512, 512, 3), dtype=np.uint8)
    test = gsoup.draw_text_on_image(test, np.array(["target"]))
    assert np.any(test) > 0
    color_image = np.random.uniform(size=(512, 512, 3))
    gray_image = gsoup.color_to_gray(color_image)
    assert gray_image.shape == (512, 512, 1)
    gray_image = gsoup.color_to_gray(color_image, keep_channels=True)
    assert gray_image.shape == (512, 512, 3)
    assert np.all(gray_image[:, :, 0] == gray_image[:, :, 1])
    assert np.all(gray_image[:, :, 1] == gray_image[:, :, 2])


# def test_video():
#     import platform
#     import os

#     if platform.system() == "Windows":
#         FFMPEG_DIR = os.path.join("C:/tools/ffmpeg-7.0.1-essentials_build/bin")
#         os.environ["PATH"] = FFMPEG_DIR + ";" + os.environ["PATH"]
#     else:
#         FFMPEG_DIR = os.path.join("/usr/bin")
#         os.environ["PATH"] = FFMPEG_DIR + ":" + os.environ["PATH"]
#     frame_number = 100
#     h = 128
#     w = 128
#     # images = np.random.randint(0, 255, (frame_number, 512, 512, 3), dtype=np.uint8)
#     images = np.array(
#         [gsoup.generate_concentric_circles(h, w, n=5) / 255.0 for i in range(100)]
#     )
#     # im1 = gsoup.generate_voronoi_diagram(512, 512, 1000)
#     # im2 = gsoup.generate_voronoi_diagram(512, 512, 1000)
#     # im1s = np.tile(im1[None, ...], (10, 1, 1, 1))
#     # im2s = np.tile(im2[None, ...], (10, 1, 1, 1))
#     # images = np.vstack([im1s, im2s])
#     gsoup.save_video(images, Path("resource/lossless_video.avi"), lossy=False, fps=10)
#     gsoup.save_video(images, Path("resource/lossy_video.avi"), lossy=True, fps=10)
#     gsoup.trim_video(
#         Path("resource/lossless_video.avi"),
#         Path("resource/trimmed.avi"),
#         0,
#         100,
#         True,
#     )
#     reader = gsoup.VideoReader(
#         Path("resource/lossless_video.avi"), h=h, w=w, verbose=True
#     )
#     fps = gsoup.FPS()
#     reader_has_frames = False
#     for i, frame in enumerate(reader):
#         reader_has_frames = True
#         print("{}: {}, fps: {}".format(i, frame.shape, fps()))
#         assert np.all(frame == images[i])
#     assert reader_has_frames
#     video_frames = gsoup.load_video(Path("resource/lossless_video.avi"))
#     assert video_frames.shape == (frame_number, h, w, 3)
#     assert np.all(video_frames == images)
#     video_frames_reversed = gsoup.reverse_video(Path("resource/lossless_video.avi"))
#     assert (video_frames_reversed[-1] == video_frames[0]).all()
#     sliced_frames = gsoup.slice_from_video(
#         Path("resource/lossless_video.avi"),
#         every_n_frames=2,
#         start_frame=0,
#         end_frame=6,
#     )
#     assert (sliced_frames == video_frames[:7:2, :, :, :]).all()
#     gsoup.video_to_images(
#         Path("resource/lossless_video.avi"),
#         Path("resource/ffmpeg_reconstructed_images"),
#     )
#     gsoup.save_video(
#         Path("resource/ffmpeg_reconstructed_images"),
#         Path("resource/reconst_lossy.avi"),
#         fps=10,
#         lossy=True,
#     )
#     gsoup.save_video(
#         Path("resource/ffmpeg_reconstructed_images"),
#         Path("resource/reconst_lossless.avi"),
#         fps=10,
#         lossy=False,
#     )
#     discrete_images = gsoup.load_images(Path("resource/ffmpeg_reconstructed_images"))
#     assert discrete_images.shape == (frame_number, h, w, 3)
#     timestamps = gsoup.get_frame_timestamps(Path("resource/lossless_video.avi"))
#     assert timestamps[0] == 0


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
    patterns = gsoup.load_images(Path("tests/tests_resource/correspondence"))
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


def test_sphere_tracer():
    image_size = 512
    # device = "cuda:0"
    device = "cpu"
    w2v, v2c = gsoup.create_random_cameras_on_unit_sphere(
        5, 1.0, opengl=True, device=device
    )
    ray_origins, ray_directions = gsoup.generate_rays(
        w2v, v2c[0], image_size, image_size, device=device
    )
    sdf = gsoup.structures.sphere_sdf(0.25)
    images = []
    for o, d in zip(ray_origins, ray_directions):
        result = gsoup.render_sdf(sdf, o.view(-1, 3), d.view(-1, 3))
        images.append(result.view(image_size, image_size, 4))
    images = gsoup.to_np(torch.stack(images))
    images = gsoup.alpha_compose(images)
    gizmo_images = gsoup.draw_gizmo_on_image(
        images, gsoup.to_np(v2c @ w2v), opengl=True
    )
    gsoup.save_images(gizmo_images, Path("resource/sphere_trace"))


def test_geometry():
    faces_tri = np.array([[0, 1, 2], [2, 3, 0]])  # Triangle example
    faces_quad = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])  # Quad example
    _, faces_quadcube = gsoup.structures.quad_cube()
    _, faces_tricube = gsoup.structures.cube()

    e, f2e, _ = gsoup.faces2edges_naive(faces_tri)
    e, f2e, _ = gsoup.faces2edges_naive(faces_tricube)
    e, f2e, _ = gsoup.faces2edges_naive(faces_quad)
    e, f2e, _ = gsoup.faces2edges_naive(faces_quadcube)


def test_rasterizer():
    # prep "canvas"
    width, height = 256, 256
    f = 800
    V3, F3 = gsoup.load_mesh("tests/tests_resource/cube.obj")  # (n, 3)
    V4, F4 = gsoup.structures.quad_cube()  # (n, 4)
    # Some random transformation
    V3 = V3 / 8
    V4 = V4 / 8
    rand_qvec = gsoup.random_qvec(1)
    rand_rot_mat = gsoup.qvec2mat(rand_qvec)
    rand_rot_trans = np.random.uniform(-0.1, 0.1, size=3)
    random_rigid = gsoup.compose_rt(
        rand_rot_mat, rand_rot_trans[None, ...], square=False
    )
    V3 = (random_rigid[0] @ gsoup.to_hom(V3).T).T
    V4 = (random_rigid[0] @ gsoup.to_hom(V4).T).T
    # Camera intrinsics matrix
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    K = K.astype(np.float32)
    # Camera extrinsics
    # Rt = np.eye(4)[:3, :]  # (identity for now)
    cam_loc = np.array([1.0, 0.0, 0.0])
    cam_at = np.array([0.0, 0.0, 0.0])
    cam_up = np.array([0.0, 0.0, 1.0])
    Rt = gsoup.look_at_np(cam_loc, cam_at, cam_up)  # cam -> world
    Rt = gsoup.invert_rigid(Rt)[0]  # world -> cam
    Rt = Rt.astype(np.float32)
    # Define colors per face
    colors3 = np.random.randint(0, 256, size=len(F3) * 3).reshape(len(F3), 3)
    colors4 = np.random.randint(0, 256, size=len(F4) * 3).reshape(len(F4), 3)
    colors1 = np.array([255, 255, 255])

    image = gsoup.render_mesh(V3, F3, K, Rt, colors3, wh=(width, height))
    gsoup.save_image(image, "resource/rast_tricube.png")

    image = gsoup.render_mesh(V4, F4, K, Rt, colors4, wh=(width, height))
    gsoup.save_image(image, "resource/rast_quadcube.png")

    image = gsoup.render_mesh(
        V4,
        F4,
        K,
        Rt,
        colors1,
        wireframe=True,
        wh=(width, height),
    )
    gsoup.save_image(image, "resource/rast_wireframe.png")

    image = gsoup.render_mesh(
        V4,
        F4,
        K,
        Rt,
        colors1,
        wireframe=True,
        wireframe_occlude=True,
        wh=(width, height),
    )
    gsoup.save_image(image, "resource/rast_wireframe_occlude.png")

    image = gsoup.render_mesh(
        V4,
        F4,
        K,
        Rt,
        colors4,
        wireframe=True,
        wh=(width, height),
    )
    gsoup.save_image(image, "resource/rast_wireframe_color.png")
    # test with providing image and depth_buffer
    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    gsoup.render_mesh(
        V4,
        F4,
        K,
        Rt,
        colors4,
        image=image,
        depth_buffer=depth_buffer,
    )
    gsoup.save_image(image, "resource/rast_provide_image.png")


def test_qem():
    v, f = gsoup.structures.cube()
    v_new, f_new = gsoup.qem(v, f, budget=4)
    assert f_new.shape[0] == 4
