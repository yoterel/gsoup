import pytest
import numpy as np
import torch
import gsoup
from pathlib import Path

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
    dst = Path("resource/voronoi.png")
    gsoup.generate_voronoi_diagram(512, 512, 1000, dst=dst)
    img = gsoup.load_image(dst)
    assert img.shape == (512, 512, 3)
    img = gsoup.load_images(dst)
    assert img.shape == (1, 512, 512, 3)
    img = gsoup.load_images([dst])
    assert img.shape == (1, 512, 512, 3)
    img = gsoup.load_images([dst, dst, dst, dst])
    assert img.shape == (4, 512, 512, 3)
    resized_img = gsoup.resize_images_naive(img, 256, 256, mode="mean")
    assert resized_img.shape == (4, 256, 256, 3)
    grid = gsoup.image_grid(resized_img, 2, 2)
    assert grid.shape == (512, 512, 3)

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


def test_qem():
    v, f = gsoup.structures.cube()
    v_new, f_new = gsoup.qem(v, f, budget = 4)
    assert f_new.shape[0] == 4