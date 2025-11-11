import subprocess
import collections
import numpy as np
from pathlib import Path
import json
import time


def get_ffmpeg_version(verbose=False):
    """Get the version of ffmpeg installed on the system.

    Args:
        verbose (bool, optional): If True, prints warning messages when ffmpeg
            is not found or version cannot be detected. Defaults to False.

    Returns:
        str or None: The ffmpeg version string if found, None otherwise.
    """
    try:
        ffmpeg_output = subprocess.run(
            [str(Path("ffmpeg")), "-version"],
            capture_output=True,
            text=True,
        ).stdout
        parts = ffmpeg_output.split()
        version = parts[parts.index("version") + 1]
        return version
    except ValueError:
        if verbose:
            print("gsoup warning: could not detect a valid ffmpeg version.")
    except FileNotFoundError:
        if verbose:
            print("gsoup warning: ffmpeg not found.")


get_ffmpeg_version(True)


class FPS:
    """Calculates and tracks frames per second (FPS) in real-time.

    This class maintains a rolling average of frame timestamps to calculate
    the current FPS. Useful for performance monitoring in video processing
    applications.

    Reference: https://stackoverflow.com/a/54539292

    Example:
        fps = FPS()
        while True:
            # Process frame
            current_fps = fps()
            print(f"Current FPS: {current_fps}")

    Attributes:
        frametimestamps (collections.deque): Circular buffer storing frame
            timestamps for FPS calculation.
    """

    def __init__(self, avarageof=50):
        """Initialize the FPS calculator.

        Args:
            avarageof (int, optional): Number of recent frames to average
                for FPS calculation. Defaults to 50.
        """
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        """Calculate and return the current FPS.

        This method should be called once per frame to update the FPS
        calculation. It adds the current timestamp and returns the
        calculated FPS based on recent frame timestamps.

        Returns:
            float: The current FPS value. Returns 0.0 if insufficient
                data is available for calculation.
        """
        self.frametimestamps.append(time.time())
        nominator = len(self.frametimestamps)
        denominator = self.frametimestamps[-1] - self.frametimestamps[0]
        if (len(self.frametimestamps) > 1) and (denominator != 0):
            return nominator / denominator
        else:
            return 0.0


def probe_video(video_path):
    """Extract video metadata using ffprobe.

    Uses ffprobe to analyze a video file and extract comprehensive metadata
    including dimensions, frame rate, codec information, and other properties.

    Args:
        video_path (str or Path): Path to the video file to analyze.

    Returns:
        dict: Dictionary containing video metadata with the following keys:
            - width (int): Video width in pixels
            - height (int): Video height in pixels
            - frame_rate (float): Frames per second
            - codec (str): Video codec name
            - pix_fmt (str): Pixel format
            - bit_rate (str): Bit rate
            - nb_frames (int): Total number of frames
            - duration (str): Video duration in seconds

    Raises:
        ValueError: If the video file cannot be processed or metadata
            extraction fails.
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
    """Extract timestamps for each frame in a video.

    Uses ffprobe to get the precise timestamp for each frame in the video,
    which can be useful for frame-accurate operations.

    Args:
        video_path (str or Path): Path to the video file.

    Returns:
        list[float]: List of timestamps in seconds for each frame.
            Invalid or unparseable timestamps are skipped.
    """
    video_path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
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
        line = line.strip(",")
        try:
            timestamps.append(float(line))
        except ValueError:
            continue
    return timestamps


def load_video(video_path):
    """Load an entire video file into memory as a numpy array.

    Reads a video file and converts it to a numpy array containing all
    frames. This is memory-intensive for large videos but provides
    fast random access to any frame.

    Args:
        video_path (str or Path): Path to the video file to load.

    Returns:
        np.ndarray: 4D array with shape (n_frames, height, width, 3)
            containing RGB pixel data for all frames.

    Raises:
        ValueError: If video dimensions cannot be determined.
    """
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
    verbose=False,
):
    """Save video frames to disk as an MP4 file.

    Writes a sequence of video frames to disk using ffmpeg. Only supports
    MP4 container format. Can handle both numpy arrays and iterables of frames.

    Args:
        frames (np.ndarray or iterable): Video frames to save. Can be:
            - numpy array with shape (n_frames, height, width, 3)
            - iterable yielding frames with shape (height, width, 3)
        output_path (str or Path): Destination path for the output video file.
            Must have .mp4 extension.
        fps (float): Frames per second for the output video.
        bitrate (str, optional): Video bitrate (e.g., "500k", "2M").
            If None, uses default bitrate.
        codec (str, optional): Video codec to use. Defaults to "libx264".
        pixel_format (str, optional): Output pixel format. Defaults to "yuv420p".
        verbose (bool, optional): If True, shows ffmpeg output. Defaults to False.

    Raises:
        ValueError: If output_path does not have .mp4 extension.
    """
    output_path = Path(output_path)
    if output_path.suffix != ".mp4":
        raise ValueError("Only mp4 container is supported.")
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
    if not verbose:
        cmd.extend(["-hide_banner", "-loglevel", "quiet"])
    cmd.append(str(output_path))
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frame_iter:
        pipe.stdin.write(frame.astype(np.uint8).tobytes())
    pipe.stdin.close()
    pipe.wait()


def reverse_video(
    input_path, output_path, bitrate=None, codec="libx264", pixel_format="yuv420p"
):
    """Reverse a video file by playing frames in reverse order.

    Creates a new video file where all frames are played in reverse order
    using ffmpeg's reverse filter.

    Args:
        input_path (str or Path): Path to the input video file.
        output_path (str or Path): Path for the reversed output video.
        bitrate (str, optional): Video bitrate for the output (e.g., "500k").
            If None, uses default bitrate.
        codec (str, optional): Video codec to use. Defaults to "libx264".
        pixel_format (str, optional): Output pixel format. Defaults to "yuv420p".

    Raises:
        subprocess.CalledProcessError: If ffmpeg command fails.
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
    """Compress a video by re-encoding it at a specified bitrate.

    Re-encodes the input video with the specified bitrate to reduce file size.
    This is useful for creating smaller versions of videos for web or storage.

    Args:
        input_path (str or Path): Path to the input video file.
        output_path (str or Path): Path for the compressed output video.
        bitrate (str): Target bitrate for compression (e.g., "500k", "1M").
        codec (str, optional): Video codec to use. Defaults to "libx264".
        pixel_format (str, optional): Output pixel format. Defaults to "yuv420p".

    Raises:
        subprocess.CalledProcessError: If ffmpeg command fails.
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
    """Extract a slice of frames from a video.

    Selects frames between start_frame and stop_frame with a given stride,
    either returning them as a numpy array or saving to a new video file.

    Args:
        input_path (str or Path): Path to the input video file.
        start_frame (int): Starting frame index (inclusive).
        stop_frame (int): Ending frame index (inclusive).
        stride (int, optional): Frame stride (step size). Defaults to 1.
        fps (float, optional): Output frame rate. If None, uses original FPS.
        codec (str, optional): Video codec for output. Defaults to "libx264".
        pixel_format (str, optional): Output pixel format. Defaults to "yuv420p".
        bitrate (str, optional): Output bitrate (e.g., "500k").
        to_numpy (bool, optional): If True, returns numpy array. If False,
            saves to output_path. Defaults to True.
        output_path (str or Path, optional): Output file path when to_numpy=False.

    Returns:
        np.ndarray or None: If to_numpy=True, returns array with shape
            (n_frames, height, width, 3). If to_numpy=False, returns None.

    Raises:
        ValueError: If video dimensions cannot be determined or output_path
            is required but not provided.
        subprocess.CalledProcessError: If ffmpeg command fails.
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
    """Extract a single frame from a video as a numpy array.

    Efficiently extracts one specific frame from a video without loading
    the entire video into memory.

    Args:
        video_path (str or Path): Path to the video file.
        frame_index (int): Index of the frame to extract (0-based).

    Returns:
        np.ndarray: 3D array with shape (height, width, 3) containing
            RGB pixel data for the specified frame.

    Raises:
        ValueError: If video info is insufficient or frame extraction fails.
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
    """Trim a video to a specific range of frames.

    Creates a new video file containing only the frames between start_frame
    and end_frame (inclusive). More efficient than loading entire video
    for simple trimming operations.

    Args:
        input_path (str or Path): Path to the input video file.
        output_path (str or Path): Path for the trimmed output video.
        start_frame (int): Starting frame index (inclusive).
        end_frame (int): Ending frame index (inclusive).
        bitrate (str, optional): Output bitrate (e.g., "500k").
        codec (str, optional): Video codec for output. Defaults to "libx264".
        pixel_format (str, optional): Output pixel format. Defaults to "yuv420p".

    Raises:
        ValueError: If frame rate information is unavailable.
        subprocess.CalledProcessError: If ffmpeg command fails.
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
    """Iterator for reading video frames one at a time.

    Provides a memory-efficient way to process video frames sequentially
    without loading the entire video into memory. Uses ffmpeg to decode
    frames on-demand.

    Example:
        reader = VideoReader("video.mp4")
        for frame in reader:
            # Process each frame
            process_frame(frame)

    Attributes:
        filename (str): Path to the video file.
        width (int): Video width in pixels.
        height (int): Video height in pixels.
        channels (int): Number of color channels (3 for RGB).
        proc (subprocess.Popen): ffmpeg process for frame decoding.
    """

    def __init__(self, filename):
        """Initialize the video reader.

        Args:
            filename (str): Path to the video file to read.
        """
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
        """Return iterator for video frames.

        Returns:
            VideoReader: Self, as this class implements the iterator protocol.
        """
        return self

    def __next__(self):
        """Get the next frame from the video.

        Returns:
            np.ndarray: 3D array with shape (height, width, channels)
                containing RGB pixel data for the next frame.

        Raises:
            StopIteration: When all frames have been read or video ends.
        """
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
