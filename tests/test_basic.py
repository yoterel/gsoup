import pytest
import numpy as np
import torch
import gsoup

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
    v = np.random.randn(100, 3)
    v_normalized = gsoup.normalize_vertices(v)
    assert (v_normalized < 1.0).all()

def test_normalize_vertices_torch():
    v = torch.randn(100, 3)
    v_normalized = gsoup.normalize_vertices(v)
    assert (v_normalized < 1.0).all()

def test_normalize_vertices_torch():
    v = torch.randn(100, 3)
    v_normalized = gsoup.normalize_vertices(v)
    assert (v_normalized < 1.0).all()