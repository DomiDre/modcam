"""Virtual Camera Module"""
import os
import time
import sys
import logging
from typing import Tuple, List
from pathlib import Path
from typing import Optional

import numpy as np
import ffmpeg


class VirtualCam:
    """
    Virtual Webcam that can be used in browsers like a normal one

    This module only works in linux operating systems and needs v4l2loopback
    installed.
    """
    def __init__(self, video_device: int, width: Optional[int] = None,
                 height: Optional[int] = None, fps: int = 30):
        """
        Initializes the virtual camera

        Args:
            video_device: On which video device shall the virtual camera be
            width: Define the image width of the camera output frames
            height: Define the image height of the camera output frames
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.video_device = video_device
        path_virtual = Path(f'/dev/video{video_device}')
        self.logger = logging.getLogger("VCam")
        self.output_queue: List[Tuple[float, np.ndarray]] = []  # list containing arrays that shall be output
        self.t_last_image_send = 0.0
        self.time_per_frame = 1 / fps

        # initialize virtual device using v4l2loopback
        if not path_virtual.exists():
            self.logger.info(
                'Couldnt find v4l2loopback device on '
                f'/dev/video{self.video_device}. '
                'Calling modprobe v4l2loopback. '
                'If you do not want the program to do it, do it yourself:')
            self.logger.info(
                'sudo modprobe v4l2loopback exclusive_caps=1 '
                f'video_nr={self.video_device} card_label="virtualcamera"')
            os.system('sudo modprobe v4l2loopback exclusive_caps=1 '
                      f'video_nr={self.video_device} card_label="virtualcamera"')
            time.sleep(1)
            if not path_virtual.exists():
                sys.exit("Could not create v4l2loopback device."
                         "Maybe try installing it following "
                         "https://github.com/umlaeute/v4l2loopback")

        # if width & height set at initialization, init ffmpeg process
        if width is not None and height is not None:
            self.init_ffmpeg()

    def init_ffmpeg(self):
        self.logger.info(f"Initializing Virtual Camera: {self.width}x{self.height} @ {self.fps} Hz")
        self.input = ffmpeg.input(
            "pipe:0",
            format="rawvideo",
            pix_fmt="rgb24",
            video_size=(self.width, self.height),
            framerate=self.fps)
        self.output = ffmpeg.output(
            self.input,
            f"/dev/video{self.video_device}",
            format="v4l2",
            vcodec="rawvideo",
            pix_fmt="yuyv422",
            framerate=self.fps,
            s="{}x{}".format(self.width, self.height))
        self.ffmpeg_proc = ffmpeg.run_async(
            self.output,
            pipe_stdin=True,
            quiet=True)

    def write(self, frame: np.ndarray) -> None:
        """Write `frame` to the virtual camera as next readable frame

        Args:
            frame: Numpy array with same width and height as passed at init in RGB format
        """
        self.ffmpeg_proc.stdin.write(frame)

    def quit(self):
        """Terminate ffmpeg process"""
        self.ffmpeg_proc.terminate()
