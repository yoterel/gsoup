import pytest
import numpy as np
import torch
import gsoup

def test_load_obj():
    v, f = gsoup.load_obj("resource/cube.obj")
    assert v.shape[0] == 8
    assert f.shape[0] == 12


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
        images.append(gsoup.render(sdf, o, d))
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

def test_qem():
    v, f = gsoup.structures.cube()
    v_new, f_new = gsoup.qem(v, f, budget = 4)
    assert f_new.shape[0] == 4