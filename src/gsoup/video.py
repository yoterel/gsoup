import time
import collections
import numpy as np
from pathlib import Path
import ffmpeg

class FPS:
    """
    calculates current fps and returns it, see https://stackoverflow.com/a/54539292
    example usage:
        fps = FPS()
        while True:
            # do something
            print(fps())
    """
    def __init__(self,avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)
    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0])
        else:
            return 0.0

def get_video_info(video_path):
    """
    returns basic video info
    :param video_path: path to video
    :return: height, width, fps
    """
    video_path = Path(video_path)
    probe = ffmpeg.probe(str(video_path))
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    fps = float(video_stream['r_frame_rate'].split('/')[0]) / float(video_stream['r_frame_rate'].split('/')[1])
    # codec_name = video_stream['codec_name']
    # format_name = probe['format']['format_name']
    return height, width, fps

def load_video(video_path):
    """
    loads a video from disk into a numpy tensor (uint8, channels last, RGB)
    :param video_path: path to video
    :return: (n x h x w x 3) tensor
    """
    h, w, _ = get_video_info(video_path)
    out, _ = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    return video

def save_video(frames, output_path, fps, lossy=False):
    """
    saves a video from a t x h x w x 3 numpy tensor
    :param frames: (t x h x w x 3) tensor
    :param output_path: path to save video to
    :param fps: frames per second of output video
    :param lossy: if True, use lossy compression (default: False, but then only .avi is supported)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if lossy:
        (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(frames.shape[2], frames.shape[1]), r=fps)
        .output(str(output_path), pix_fmt='yuv420p')
        .overwrite_output()
        .run(input=frames.tobytes())
        )
    else:
        if output_path.suffix == ".avi":
            pix_fmt = "bgr24"
        else:
            # todo: figure out lossless pixel formats for other containers
            raise ValueError("Lossless video only supported for .avi container")
        (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(frames.shape[2], frames.shape[1]), r=fps)
            .output(str(output_path), vcodec='rawvideo', pix_fmt=pix_fmt)
            .overwrite_output()
            .run(input=frames.tobytes())
        )

def reverse_video(video_path, output_path=None):
    """
    reverses a video and possibly saves it to disk
    :param video_path: path to video
    :param output_path: path to save video to
    :return: (n x h x w x 3) tensor of reversed video
    """
    h, w, _ = get_video_info(video_path)
    out, _ = (
        ffmpeg
        .input(video_path)
        .filter('reverse')
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    if output_path is not None:
        save_video(video, output_path)
    return video