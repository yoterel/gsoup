import time
import collections
import numpy as np
import pathlib
from pathlib import Path
import ffmpeg
import subprocess
import cv2

# import os
# FFMPEG_PATH = ""


def get_ffmpeg_version(verbose=False):
    """
    :return: ffmpeg version
    """
    # global FFMPEG_PATH
    FFMPEG_PATH = ""
    version = None
    # env_var = os.environ.get("FFMPEG_PATH")
    # if env_var:
    #     FFMPEG_PATH = env_var
    try:
        ffmpeg_output = subprocess.run(
            [str(Path(FFMPEG_PATH, "ffmpeg")), "-version"],
            capture_output=True,
            text=True,
        ).stdout
        parts = ffmpeg_output.split()
        version = parts[parts.index("version") + 1]
        if verbose:
            if int(version[0]) < 5:
                print(
                    "gsoup warning: detected ffmpeg version: {}. Video module May fail for versions lower than 5".format(
                        version
                    )
                )
                # print(
                #     "you can set the environment variable "
                #     "FFMPEG_PATH"
                #     " to the path of ffmpeg binary"
                # )
    except ValueError:
        if verbose:
            print(
                "gsoup warning: could not detect ffmpeg version. Video module May fail"
            )
            # print(
            #     "you can set the environment variable "
            #     "FFMPEG_PATH"
            #     " to the path of ffmpeg binary"
            # )
    except FileNotFoundError:
        if verbose:
            print("gsoup warning: ffmpeg not found. Video module May fail")
            # print(
            #     "you can set the environment variable "
            #     "FFMPEG_PATH"
            #     " to the path of ffmpeg binary"
            # )
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


def get_video_info(video_path):
    """
    returns basic video info
    :param video_path: path to video
    :return: height, width, fps, frame_count
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError("Video file not found: {}".format(video_path))
    probe = ffmpeg.probe(str(video_path))
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    fps = float(video_stream["r_frame_rate"].split("/")[0]) / float(
        video_stream["r_frame_rate"].split("/")[1]
    )
    frame_count = int(video_stream["nb_frames"])
    info = {
        "height": height,
        "width": width,
        "pixel_format": video_stream["pix_fmt"],
        "fps": fps,
        "frame_count": frame_count,
        "codec_name": video_stream["codec_name"],
        "bit_rate": video_stream["bit_rate"],
    }
    return info


def get_frame_timestamps(video_path):
    """
    returns numpy array of frame timestamps in seconds for a video
    :param video_path: path to video
    :return: numpy array of frame timestamps in seconds
    """
    video_path = Path(video_path)
    probe = ffmpeg.probe(str(video_path), show_frames="-show_frames")
    video_frames = [
        frame for frame in probe["frames"] if frame["media_type"] == "video"
    ]
    frame_times = np.array([float(frame["pts_time"]) for frame in video_frames])
    return frame_times


def load_video(video_path, verbose=False):
    """
    loads a video from disk into a numpy array (uint8, channels last, RGB)
    :param video_path: path to video
    :return: (n x h x w x 3) numpy array
    """
    info = get_video_info(video_path)
    out, _ = (
        ffmpeg.input(str(video_path))
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, quiet=not verbose)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, info["height"], info["width"], 3])
    return video


def save_video(frames, output_path, fps, bit_rate="1M", lossy=True, verbose=False):
    """
    saves a video from a t x h x w x 3 numpy array
    :param frames: (t x h x w x 3) numpy array or directory path containing images of same format and resolution
    :param output_path: path to save video to
    :param fps: frames per second of output video
    :param lossy: if true, use lossy compression (if False, only .avi is supported)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if type(frames) == np.ndarray:
        if lossy:
            (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s="{}x{}".format(frames.shape[2], frames.shape[1]),
                    r=fps,
                )
                .output(str(output_path), pix_fmt="yuv420p", b=bit_rate)
                .overwrite_output()
                .run(input=frames.tobytes(), quiet=not verbose)
            )
        else:
            if output_path.suffix == ".avi":
                pix_fmt = "bgr24"
            else:
                # todo: figure out lossless pixel formats for other containers
                raise ValueError("Lossless video only supported for .avi container")
            (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s="{}x{}".format(frames.shape[2], frames.shape[1]),
                    r=fps,
                )
                .output(str(output_path), vcodec="rawvideo", pix_fmt=pix_fmt)
                .overwrite_output()
                .run(input=frames.tobytes(), quiet=not verbose)
            )
    elif isinstance(frames, pathlib.PurePath):
        if not frames.exists():
            raise FileNotFoundError("Image directory not found: {}".format(frames))
        if not frames.is_dir():
            raise ValueError(
                "frames must be numpy array or path to directory of images"
            )
        files = sorted(list(frames.glob("*")))
        if len(files) == 0:
            raise FileNotFoundError("No images found in directory: {}".format(frames))
        files = [x for x in files if x.is_file()]
        files = [x for x in files if x.suffix in [".png", ".jpg", ".jpeg", ".bmp"]]
        extensions = [file.suffix for file in files]
        if len(set(extensions)) > 1:
            raise ValueError("All images in directory must have same extension")
        with open("ffmpeg_input.txt", "wb") as outfile:
            for filename in files:
                mystr = "file '{}'\n".format(str(filename.resolve()).replace("\\", "/"))
                outfile.write(mystr.encode())
                mystr = "duration {:.05f}\n".format(1 / fps)
                outfile.write(mystr.encode())
        stdin_stream = None
        stdout_stream = subprocess.PIPE if not verbose else None
        stderr_stream = subprocess.PIPE if not verbose else None
        if lossy:
            args = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                "ffmpeg_input.txt",
                "-b",
                bit_rate,
                "-r",
                str(fps),
                str(output_path),
            ]
        else:
            if output_path.suffix == ".avi":
                pix_fmt = "bgr24"
            else:
                # todo: figure out lossless pixel formats for other containers
                raise ValueError("Lossless video only supported for .avi container")
            args = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                "ffmpeg_input.txt",
                "-r",
                str(fps),
                "-pix_fmt",
                pix_fmt,
                "-vcodec",
                "rawvideo",
                str(output_path),
            ]
        proc = subprocess.Popen(
            args, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream
        )
        out, err = proc.communicate(input)
        retcode = proc.poll()
        if retcode:
            raise ValueError("ffmpeg", out, err)
        Path("ffmpeg_input.txt").unlink()
    else:
        raise ValueError("frames must be numpy array or path to directory of images")


def reverse_video(video_path, output_path=None, verbose=False):
    """
    reverses a video and optionally saves it to disk
    :param video_path: path to video
    :param output_path: path to save video to
    :return: (n x h x w x 3) tensor of reversed video
    """
    info = get_video_info(video_path)
    out, _ = (
        ffmpeg.input(str(video_path))
        .filter("reverse")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, quiet=not verbose)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, info["height"], info["width"], 3])
    if output_path is not None:
        save_video(video, output_path)
    return video


def compress_video(src, dst, crf=23, verbose=False):
    """
    compresses a video using ffmpeg & libx264, saving it to disk
    :param src: path to video
    :param dst: path to save compressed video to
    :param crf: constant rate factor (lower is better quality, 0-51, 0 is lossless)
    :param verbose: if True, print ffmpeg output
    """
    if not (0 <= crf <= 51):
        raise ValueError("crf must be between 0 and 51")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg.input(str(src))
        .output(str(dst), vcodec="libx264", crf=crf, preset="slow")
        .overwrite_output()
        .run(quiet=not verbose)
    )


def images_to_video(
    src: Path, dst: Path, fps, bit_rate="1M", lossy=True, verbose=False
):
    """
    creates a video from a folder of images
    :param src: path to images
    :param dst: path to save video to
    :param fps: frames per second of output video
    :param bit_rate: bit rate of output video
    :param lossy: if true, use lossy compression (if False, only .avi is supported)
    :param verbose: if True, print ffmpeg output
    """
    save_video(src, dst, fps=fps, bit_rate=bit_rate, lossy=lossy, verbose=verbose)


def video_to_images(src, dst, verbose=False):
    """
    creates a folder of images from a video
    :param src: path to video
    :param dst: path to save images to (images will be named 0000.png, 0001.png, etc.)
    :param verbose: if True, print ffmpeg output
    """
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg.input(str(src))
        .output(str(dst / "%04d.png"), vcodec="png", format="image2")
        .overwrite_output()
        .run(quiet=not verbose)
    )


def slice_from_video(
    src, every_n_frames=2, start_frame=0, end_frame=None, verbose=False
):
    """
    slices a video into frames, returning a numpy array of frames
    :param src: path to video
    :param every_n_frames: stride between selected frames
    :param start_frame: first frame to take (inclusive)
    :param end_frame: last frame to take (inclusive)
    :return: (n x h x w x 3) tensor of sliced video
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError("Video file not found: {}".format(src))
    info = get_video_info(src)
    h, w, fc = info["height"], info["width"], info["frame_count"]
    if end_frame is None:
        end_frame = fc + 1
    out, _ = (
        ffmpeg.input(str(src))
        .trim(start_frame=start_frame, end_frame=end_frame + 1)
        .filter("select", "not(mod(n,{}))".format(every_n_frames))
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", fps_mode="passthrough")
        .run(capture_stdout=True, quiet=not verbose)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    return video


def get_frame_from_video(src, frame_index, verbose=False):
    """
    gets a single frame from a video
    :param src: path to video
    :param frame_num: frame number to get
    :return: (h x w x 3) tensor of frame
    """
    info = get_video_info(src)
    out, _ = (
        ffmpeg.input(str(src))
        .filter("select", "eq(n,{})".format(frame_index))
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", fps_mode="passthrough")
        .run(capture_stdout=True, quiet=not verbose)
    )
    frame = np.frombuffer(out, np.uint8).reshape([info["height"], info["width"], 3])
    return frame


def trim_video(src, dst, start_frame=0, end_frame=None, verbose=False):
    """
    trim a video from disk and save it to disk
    note: will reencode video
    :param src: path to video
    :param dst: path to save trimmed video to
    :param start_frame: first frame to take (inclusive)
    :param end_frame: last frame to take (inclusive)
    :param verbose: if True, print ffmpeg output
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError("Video file not found: {}".format(src))
    info = get_video_info(src)
    if end_frame is None:
        end_frame = info["frame_count"] + 1
    out, _ = (
        ffmpeg.input(str(src))
        .trim(start_frame=start_frame, end_frame=end_frame + 1)
        .output(
            str(dst), vcodec="libx264", b=info["bit_rate"], pix_fmt=info["pixel_format"]
        )
        .overwrite_output()
        .run(capture_stdout=True, quiet=not verbose)
    )


class VideoReader:
    """
    A (very) basic video iterator
    """

    def __init__(
        self,
        video_path,
        h=None,
        w=None,
        every_n_frames=1,
        start_frame=0,
        end_frame=None,
        verbose=False,
    ):
        """
        :param video_path: path to video
        :param target_resolution: (h, w) tuple of target resolution (must have common divisor with original resolution)
        """
        self.video_path = Path(video_path)
        info = get_video_info(video_path)
        self.h, self.w, self.fps, self.fc = (
            info["width"],
            info["height"],
            info["fps"],
            info["frame_count"],
        )
        self.th = None
        self.tw = None
        if h is not None:
            if w is not None:
                self.th, self.tw = h, w
            else:
                raise ValueError("if h or w are specified, both must be specified")
        self.start_frame = start_frame
        if end_frame is None:
            self.end_frame = self.fc + 1
        else:
            self.end_frame = end_frame
        self.every_n_frames = every_n_frames
        self.stream = (
            ffmpeg.input(str(self.video_path))
            .trim(start_frame=self.start_frame, end_frame=self.end_frame)
            .filter("select", "not(mod(n,{}))".format(self.every_n_frames))
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", fps_mode="passthrough")
            .run_async(pipe_stdout=True, quiet=not verbose)
        )

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.fc:
            in_bytes = self.stream.stdout.read(self.h * self.w * 3)
            if not in_bytes:  # should never happen
                self.stream.stdout.close()
                raise StopIteration
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.h, self.w, 3])
            if self.th is not None:
                if self.th != self.h or self.tw != self.w:
                    frame = cv2.resize(frame, (self.tw, self.th))
            self.n += 1
            return frame
        else:
            in_bytes = self.stream.stdout.read(self.h * self.w * 3)
            assert not in_bytes  # sanity check, no bytes should be left
            self.stream.stdout.close()
            raise StopIteration
