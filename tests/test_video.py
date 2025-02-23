import pytest
import os
import numpy as np
import gsoup
from pathlib import Path


@pytest.fixture
def dummy_video():
    """
    Create a dummy video file with 10 frames (64x64) where each frame is filled
    with a constant value that increases with the frame index.
    """
    n_frames = 10
    height, width = 64, 64
    frames = []
    for i in range(n_frames):
        # Create a frame with a constant value; values increase with frame index.
        frame = np.full((height, width, 3), fill_value=i * 25, dtype=np.uint8)
        frames.append(frame)
    frames_np = np.array(frames)
    video_path = Path("resource", "dummy.mp4")
    # Write video at 10 fps with a given bitrate.
    gsoup.save_video(frames_np, video_path, fps=10, bitrate="500k")
    return video_path, frames_np


@pytest.fixture
def dummy_video_long():
    """
    Create a dummy video file with 10 frames (64x64) where each frame is filled
    with a constant value that increases with the frame index.
    """
    n_frames = 1000
    height, width = 64, 64
    frames = []
    for i in range(n_frames):
        # Create a frame with a constant value; values increase with frame index.
        frame = np.random.randint(0, high=256, size=(height, width, 3))
        frames.append(frame)
    frames_np = np.array(frames)
    video_path = Path("resource", "dummy_long.mp4")
    # Write video at 10 fps with a given bitrate.
    gsoup.save_video(frames_np, video_path, fps=10, bitrate="500k")
    return video_path, frames_np


def test_probe_video(dummy_video):
    video_path, _ = dummy_video
    info = gsoup.probe_video(video_path)
    # Check that width and height are detected correctly.
    assert info["width"] == 64
    assert info["height"] == 64
    # The number of frames might be approximate due to encoding;
    # ensure we have at least most of them.
    assert int(info["nb_frames"]) >= 8


def test_load_video(dummy_video):
    video_path, frames_np = dummy_video
    # Test reading all frames at once.
    frames = gsoup.load_video(video_path)
    assert frames.ndim == 4
    assert len(frames) == frames_np.shape[0]
    assert frames.shape[1] == 64
    assert frames.shape[2] == 64
    # Test streaming version.
    # frames_stream = list(gsoup.load_video(video_path, stream=True))


def test_get_frame(dummy_video):
    video_path, frames_np = dummy_video
    # Extract the first frame.
    frame = gsoup.get_single_frame(video_path, 0)
    assert frame.shape == (64, 64, 3)
    # Due to potential compression artifacts, allow a small tolerance.
    diff = np.abs(frame.astype(np.int32) - frames_np[0].astype(np.int32))
    assert np.mean(diff) < 20


def test_reverse_video(dummy_video):
    video_path, frames_np = dummy_video
    reversed_video_path = Path("resource", "reversed.mp4")
    gsoup.reverse_video(video_path, reversed_video_path, bitrate="500k")
    # Read frames from the reversed video.
    rev_frames = gsoup.load_video(reversed_video_path)
    # Compare with the original frames in reverse order.
    original_reversed = frames_np[::-1]
    min_frames = min(len(rev_frames), len(original_reversed))
    diff = np.abs(
        rev_frames[:min_frames].astype(np.int32)
        - original_reversed[:min_frames].astype(np.int32)
    )
    assert np.mean(diff) < 25


def test_compress_video(dummy_video):
    video_path, _ = dummy_video
    compressed_video_path = Path("resource", "compressed.mp4")
    gsoup.compress_video(video_path, compressed_video_path, bitrate="100k")
    assert compressed_video_path.exists()
    # Optionally, compare file sizes (compressed should be smaller).
    orig_size = os.path.getsize(video_path)
    comp_size = os.path.getsize(compressed_video_path)
    assert comp_size < orig_size


def test_slice_video(dummy_video):
    video_path, _ = dummy_video
    # Slice frames from 2 to 8 with a stride of 2 (expecting frames 2, 4, 6, 8).
    sliced_frames = gsoup.slice_video(
        video_path,
        start_frame=2,
        stop_frame=8,
        stride=2,
        fps=10,
        bitrate="500k",
    )
    assert sliced_frames.shape[0] == 4
    sliced_video_path = Path("resource", "sliced.mp4")
    gsoup.slice_video(
        video_path,
        start_frame=2,
        stop_frame=8,
        stride=2,
        fps=10,
        bitrate="500k",
        to_numpy=False,
        output_path=sliced_video_path,
    )
    sliced_frames = gsoup.load_video(sliced_video_path)
    assert sliced_frames.shape[0] == 4


def test_trim_video(dummy_video):
    video_path, _ = dummy_video
    # Trim video from frame 3 to frame 7 (should contain ~4 frames).
    trimmed_video_path = Path("resource", "trimmed.mp4")
    gsoup.trim_video(
        video_path, trimmed_video_path, start_frame=3, end_frame=7, bitrate="500k"
    )
    trimmed_frames = gsoup.load_video(trimmed_video_path)
    # Allow a one-frame tolerance.
    assert abs(trimmed_frames.shape[0] - 4) <= 1


def test_get_frame_timestamps(dummy_video):
    video_path, _ = dummy_video
    timestamps = gsoup.get_frame_timestamps(video_path)
    assert np.all(np.isclose(timestamps, np.linspace(0.0, 1.0, 10, endpoint=False)))


def test_video_reader(dummy_video_long):
    video_path, frames_np = dummy_video_long
    reader = gsoup.VideoReader(video_path)
    # fps = gsoup.FPS()
    frames = []
    for frame in reader:
        frames.append(frame)
        # print(fps())
    assert len(frames) == frames_np.shape[0]
