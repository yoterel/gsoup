import pytest
import numpy as np
import torch
import gsoup
from pathlib import Path

def test_rotations():
    qvecs = gsoup.random_qvec(10)
    torch_qvecs = torch.tensor(qvecs)
    rotmats = gsoup.batch_qvec2mat(qvecs)
    assert rotmats.shape == (10, 3, 3)
    rotmats = gsoup.batch_qvec2mat(torch_qvecs)
    assert rotmats.shape == (10, 3, 3)
    new_qvecs = gsoup.batch_mat2qvec(rotmats)
    mask1 = (torch.abs(new_qvecs - torch_qvecs) < 1e-6)
    mask2 = (torch.abs(new_qvecs + torch_qvecs) < 1e-6)
    assert torch.all(mask1 | mask2)
    rotmat = gsoup.qvec2mat(torch_qvecs[0])
    assert rotmat.shape == (3, 3)
    new_qvec = gsoup.mat2qvec(rotmat)
    mask1 = (torch.abs(new_qvec - torch_qvecs[0]) < 1e-6)
    mask2 = (torch.abs(new_qvec + torch_qvecs[0]) < 1e-6)
    assert torch.all(mask1 | mask2)
    normal = torch.tensor([0, 0, 1.0])
    random_vectors = gsoup.random_vectors_on_hemisphere(10, normal=normal)
    assert random_vectors.shape == (10, 3)
    assert (random_vectors @ normal).all() > 0
    normal = torch.tensor([[0, 0, 1.0]]).repeat(10, 1)
    random_vectors = gsoup.random_vectors_on_hemisphere(10, normal=normal)
    assert random_vectors.shape == (10, 3)
    assert (random_vectors[:, None, :] @ normal[:, :, None]).all() > 0

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

def test_sphere_tracer():
    w2v, v2c = gsoup.create_random_cameras_on_unit_sphere(4, 1.0, "cuda:0")
    ray_origins, ray_directions = gsoup.generate_rays(w2v, v2c, 512, 512, "cuda:0")
    sdf = gsoup.structures.sphere_sdf(0.5)
    images = []
    for o, d in zip(ray_origins, ray_directions):
        result = gsoup.render(sdf, o.view(-1, 3), d.view(-1, 3))
        images.append(result.view(512, 512, 4))
    images = torch.stack(images)

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

def test_compose_rt():
    R = np.random.randn(2, 3, 3)
    t = np.random.randn(1, 3)
    Rt = gsoup.compose_rt(R, t)
    assert Rt.shape == (2, 3, 4)

def test_normalize_vertices_np():
    v = np.random.rand(100, 3) * 100
    v_normalized = gsoup.normalize_vertices(v)
    assert (v_normalized < 1.0).all()

def test_normalize_vertices_torch():
    v = torch.rand(100, 3) * 100
    v_normalized = gsoup.normalize_vertices(v)
    assert (v_normalized < 1.0).all()

def test_structures():
    v, f = gsoup.structures.cube()
    gsoup.save_obj("resource/cube.obj", v, f)
    v1, f1 = gsoup.load_obj("resource/cube.obj")
    assert np.allclose(v, v1)
    assert np.allclose(f, f1)
    v, f = gsoup.structures.icosehedron()
    gsoup.save_obj("resource/ico.obj", v, f)
    v1, f1 = gsoup.load_obj("resource/ico.obj")
    assert np.allclose(v, v1)
    assert np.allclose(f, f1)
    v, f = gsoup.load_obj("resource/cube.obj")
    assert v.shape[0] == 8
    assert f.shape[0] == 12

def test_image():
    gsoup.generate_lollipop_pattern(512, 512, dst=Path("resource/lollipop.png"))
    gsoup.generate_concentric_circles(256, 512, dst=Path("resource/circles.png"))
    gsoup.generate_stripe_pattern(256, 512, direction="both", dst=Path("resource/stripe.png"))
    gsoup.generate_dot_pattern(512, 256, dst=Path("resource/dots.png"))
    gray1 = gsoup.generate_gray_gradient(256, 256, grayscale=True, dst=Path("resource/gg_vert.png"))
    assert gray1.shape == (256, 256)
    assert len(np.unique(gray1)) == 10
    assert gray1.max() == 255
    gray2 = gsoup.generate_gray_gradient(50, 800, vertical=False, dst=Path("resource/gg_horiz.png"))
    assert gray2.shape == (50, 800, 3)
    assert gray2.max() == 255
    gray3 = gsoup.generate_gray_gradient(256, 256, bins=-65, dst=Path("resource/gg_bin_min.png"))
    assert gray3.max() == 0
    gray4 = gsoup.generate_gray_gradient(256, 256, bins=300, dst=Path("resource/gg_bin_max.png"))
    assert gray4.max() == 255
    gray5 = gsoup.generate_gray_gradient(1080, 1920, bins=300, dst=Path("resource/gg_highres.png"))
    assert gray5.shape == (1080, 1920, 3)
    assert gray5.max() == 255
    dst = Path("resource/voronoi.png")
    gsoup.generate_voronoi_diagram(512, 512, 1000, dst=dst)
    img = gsoup.load_image(dst)
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.uint8
    img = gsoup.load_image(dst, as_grayscale=True)
    assert img.shape == (512, 512)
    assert img.dtype == np.uint8
    img = gsoup.load_image(dst, to_float=True)
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.float32
    assert (img>=0.0).all()
    assert (img<=1.0).all()
    img = gsoup.load_image(dst, channels_last=False)
    assert img.shape == (3, 512, 512)
    img = gsoup.load_image(dst, to_float=True, as_grayscale=True)
    assert img.shape == (512, 512)
    assert img.dtype == np.float32
    assert (img>=0.0).all()
    assert (img<=1.0).all()
    img = gsoup.load_image(dst, channels_last=False, as_grayscale=True)
    assert img.shape == (512, 512)
    img = gsoup.load_images(dst)
    assert img.shape == (1, 512, 512, 3)
    img = gsoup.load_images(dst, as_grayscale=True)
    assert img.shape == (1, 512, 512)
    img = gsoup.load_images([dst])
    assert img.shape == (1, 512, 512, 3)
    img = gsoup.load_images([dst, dst, dst, dst])
    assert img.shape == (4, 512, 512, 3)
    resized_img = gsoup.resize_images_naive(img, 256, 256, mode="mean")
    assert resized_img.shape == (4, 256, 256, 3)
    grid = gsoup.image_grid(resized_img, 2, 2)
    assert grid.shape == (512, 512, 3)
    img = gsoup.load_images([dst, dst, dst, dst], as_grayscale=True)
    assert img.shape == (4, 512, 512)
    img = gsoup.load_images([dst, dst, dst, dst], as_grayscale=True, channels_last=False)
    assert img.shape == (4, 512, 512)
    img = gsoup.load_images([dst, dst, dst, dst], resize_wh=(128, 128))
    assert img.shape == (4, 128, 128, 3)
    img, paths = gsoup.load_images([dst, dst, dst, dst], resize_wh=(128, 256), as_grayscale=True, channels_last=False, return_paths=True, to_float=True, to_torch=True)
    assert len(paths) == 4
    assert img.dtype == torch.float32
    assert img.shape == (4, 256, 128)

def test_video():
    frame_number = 100
    images = np.random.randint(0, 255, (frame_number, 512, 512, 3), dtype=np.uint8)
    # im1 = gsoup.generate_voronoi_diagram(512, 512, 1000)
    # im2 = gsoup.generate_voronoi_diagram(512, 512, 1000)
    # im1s = np.tile(im1[None, ...], (10, 1, 1, 1))
    # im2s = np.tile(im2[None, ...], (10, 1, 1, 1))
    # images = np.vstack([im1s, im2s])
    dst = Path("resource/noise.avi")
    gsoup.save_video(images, dst, fps=10)
    reader = gsoup.VideoReader(dst, h=512, w=512)
    fps = gsoup.FPS()
    for i, frame in enumerate(reader):
        print("{}: {}, fps: {}".format(i, frame.shape, fps()))
        assert np.all(frame == images[i])
    video_frames = gsoup.load_video(dst)
    assert video_frames.shape == (frame_number, 512, 512, 3)
    assert np.all(video_frames == images)
    video_frames_reversed = gsoup.reverse_video(dst)
    assert (video_frames_reversed[-1] == video_frames[0]).all()
    sliced_frames = gsoup.slice_from_video(dst, every_n_frames=2, start_frame=0, end_frame=6)
    assert (sliced_frames == video_frames[:7:2, :, :, :]).all()
    gsoup.video_to_images(dst, Path("resource/noise"))
    discrete_images = gsoup.load_images(Path("resource/noise"))
    assert discrete_images.shape == (frame_number, 512, 512, 3)
    timestamps = gsoup.get_frame_timestamps(dst)
    assert timestamps[0] == 0

def test_procam():
    gc_patterns = gsoup.generate_gray_code(512, 512, 1)
    # todo test more functions from this module

def test_qem():
    v, f = gsoup.structures.cube()
    v_new, f_new = gsoup.qem(v, f, budget = 4)
    assert f_new.shape[0] == 4