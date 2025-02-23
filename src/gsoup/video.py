import subprocess
import collections
import numpy as np
from pathlib import Path
import json
import time


def get_ffmpeg_version(verbose=False):
    """
    :return: ffmpeg version
    """
    try:
        ffmpeg_output = subprocess.run(
            [str(Path("ffmpeg")), "-version"],
            capture_output=True,
            text=True,
        ).stdout
        parts = ffmpeg_output.split()
        version = parts[parts.index("version") + 1]
    except ValueError:
        if verbose:
            print("gsoup warning: could not detect a valid ffmpeg version.")
    except FileNotFoundError:
        if verbose:
            print("gsoup warning: ffmpeg not found.")
    return version


get_ffmpeg_version(True)


class FPS:
    """
    calculates current fps and returns it, see https://stackoverflow.com/a/54539292
    example usage:
        fps = FPS()
        while True:
            # do something
            print(fps())
    """

    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        nominator = len(self.frametimestamps)
        denominator = self.frametimestamps[-1] - self.frametimestamps[0]
        if (len(self.frametimestamps) > 1) and (denominator != 0):
            return nominator / denominator
        else:
            return 0.0


def probe_video(video_path):
    """
    Probe the video using ffprobe to extract useful metadata.
    Returns a dict with keys: width, height, frame_rate, codec, pix_fmt, bit_rate, nb_frames, duration.
    """
    video_path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name,pix_fmt,bit_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    try:
        num, den = stream["r_frame_rate"].split("/")
        frame_rate = float(num) / float(den)
    except Exception:
        frame_rate = None
    width = stream.get("width")
    height = stream.get("height")
    codec_name = stream.get("codec_name")
    pix_fmt = stream.get("pix_fmt")
    bit_rate = stream.get("bit_rate")
    nb_frames = stream.get("nb_frames")
    duration = stream.get("duration")
    if nb_frames is None and duration is not None and frame_rate is not None:
        nb_frames = int(float(duration) * frame_rate)
    else:
        try:
            nb_frames = int(nb_frames)
        except Exception:
            nb_frames = None
    return {
        "width": width,
        "height": height,
        "frame_rate": frame_rate,
        "codec": codec_name,
        "pix_fmt": pix_fmt,
        "bit_rate": bit_rate,
        "nb_frames": nb_frames,
        "duration": duration,
    }


def get_frame_timestamps(video_path):
    """
    Get a list of timestamps (in seconds) for each frame in the video.
    """
    video_path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pkt_pts_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    lines = result.stdout.strip().splitlines()
    timestamps = []
    for line in lines:
        try:
            timestamps.append(float(line))
        except ValueError:
            continue
    return timestamps


def load_video(video_path):
    video_path = Path(video_path)
    info = probe_video(video_path)
    width = info["width"]
    height = info["height"]
    if width is None or height is None:
        raise ValueError("Could not determine video dimensions.")
    frame_size = width * height * 3  # for rgb24
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-",
    ]
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_video = pipe.stdout.read()
    pipe.stdout.close()
    pipe.wait()
    n_frames = len(raw_video) // frame_size
    video = np.frombuffer(raw_video, dtype=np.uint8).reshape(
        (n_frames, height, width, 3)
    )
    return video


def save_video(
    frames,
    output_path,
    fps,
    bitrate=None,
    codec="libx264",
    pixel_format="yuv420p",
):
    """
    Write video frames to disk.
    Parameters:
      frames: (n_frames, height, width, 3) np array or an iterable yielding such frames.
      output_path: destination video file.
      fps: frames per second.
      bitrate: optional bitrate (e.g., "500k").
      codec: video codec to use.
      pixel_format: output pixel format.
    """
    output_path = Path(output_path)
    # Determine frame dimensions
    if isinstance(frames, np.ndarray):
        height, width = frames.shape[1:3]
        frame_iter = frames
    else:
        frame_iter = iter(frames)
        first_frame = next(frame_iter)
        height, width = first_frame.shape[:2]
        frame_iter = (frame for frame in [first_frame] + list(frame_iter))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-pix_fmt",
        pixel_format,
    ]
    if bitrate:
        cmd.extend(["-b:v", str(bitrate)])
    cmd.append(str(output_path))
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frame_iter:
        pipe.stdin.write(frame.astype(np.uint8).tobytes())
    pipe.stdin.close()
    pipe.wait()


def reverse_video(
    input_path, output_path, bitrate=None, codec="libx264", pixel_format="yuv420p"
):
    """
    Reverse a video on disk.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "reverse",
        "-an",
        "-vcodec",
        codec,
        "-pix_fmt",
        pixel_format,
    ]
    if bitrate:
        cmd.extend(["-b:v", str(bitrate)])
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)


def compress_video(
    input_path, output_path, bitrate, codec="libx264", pixel_format="yuv420p"
):
    """
    Compress a video by re-encoding it at the specified bitrate.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-vcodec",
        codec,
        "-pix_fmt",
        pixel_format,
        "-b:v",
        str(bitrate),
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def slice_video(
    input_path,
    start_frame,
    stop_frame,
    stride=1,
    fps=None,
    codec="libx264",
    pixel_format="yuv420p",
    bitrate=None,
    to_numpy=True,
    output_path=None,
):
    """
    Slice a video by selecting frames between start_frame and stop_frame with a given stride.
    Outputs a new video file.
    """
    input_path = Path(input_path)
    # Build a filter chain: select frames in range and skip by stride.
    vf_expr = (
        f"select='between(n\\,{start_frame}\\,{stop_frame})*"
        f"not(mod(n-{start_frame}\\,{stride}))',setpts=N/FRAME_RATE/TB"
    )
    base_cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vf", vf_expr, "-an"]
    if fps:
        base_cmd.extend(["-r", str(fps)])
    if bitrate:
        base_cmd.extend(["-b:v", str(bitrate)])

    if to_numpy:
        # We want raw video output so we can convert to a numpy array.
        base_cmd.extend(
            ["-f", "rawvideo", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "pipe:1"]
        )
        proc = subprocess.Popen(
            base_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        info = probe_video(input_path)
        width = info["width"]
        height = info["height"]
        if width is None or height is None:
            raise ValueError("Could not determine video dimensions for slicing.")
        frame_size = width * height * 3
        raw_video = proc.stdout.read()
        proc.stdout.close()
        proc.wait()
        n_frames = len(raw_video) // frame_size
        video_array = np.frombuffer(raw_video, dtype=np.uint8).reshape(
            (n_frames, height, width, 3)
        )
        return video_array
    else:
        if output_path is None:
            raise ValueError("output_path must be provided when to_numpy is False.")
        output_path = Path(output_path)
        base_cmd.append(str(output_path))
        subprocess.run(base_cmd, check=True)


def get_single_frame(video_path, frame_index):
    """
    Extract a single frame from a video as a numpy array.
    """
    video_path = Path(video_path)
    info = probe_video(video_path)
    width = info["width"]
    height = info["height"]
    if width is None or height is None or info["frame_rate"] is None:
        raise ValueError("Insufficient video info to extract frame.")
    frame_size = width * height * 3
    fps = info["frame_rate"]
    # Calculate timestamp (in seconds) for the given frame index.
    timestamp = frame_index / fps
    cmd = [
        "ffmpeg",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_frame = result.stdout
    if len(raw_frame) < frame_size:
        raise ValueError("Could not extract frame.")
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
    return frame


def trim_video(
    input_path,
    output_path,
    start_frame,
    end_frame,
    bitrate=None,
    codec="libx264",
    pixel_format="yuv420p",
):
    """
    Trim a video using frame indices.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    info = probe_video(input_path)
    if info["frame_rate"] is None:
        raise ValueError("Frame rate information unavailable.")
    fps = info["frame_rate"]
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-an",
        "-vcodec",
        codec,
        "-pix_fmt",
        pixel_format,
    ]
    if bitrate:
        cmd.extend(["-b:v", str(bitrate)])
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)


class VideoReader:
    def __init__(self, filename):
        self.filename = filename
        info = probe_video(filename)
        self.width = info["width"]
        self.height = info["height"]
        self.channels = 3  # For 'rgb24', there are 3 channels (R, G, B)
        # Launch ffmpeg to decode video into raw video frames
        self.proc = subprocess.Popen(
            ["ffmpeg", "-i", self.filename, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # or subprocess.PIPE to capture errors
        )

    def __iter__(self):
        return self

    def __next__(self):
        frame_size = self.width * self.height * self.channels
        # Read enough bytes for one frame
        raw_frame = self.proc.stdout.read(frame_size)
        if len(raw_frame) < frame_size:
            # Clean up if we're done
            self.proc.stdout.close()
            self.proc.wait()
            raise StopIteration
        # Convert bytes to a numpy array and reshape to (height, width, channels)
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        return frame.reshape((self.height, self.width, self.channels))
