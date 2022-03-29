import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2


class Webcam:
    """Class to access physical webcam"""
    def __init__(self, width: int, height: int, fps: int, device: int = 0):
        """
        Initialize webcam using opencv

        Args:
            width (int): Target width of image requested from webcam
            height (int): Target height of image requested from webcam
            fps (int): Target FPS requested from webcam
            device (int): On which device number the webcam is read
        """
        self.path_webcam = Path(f'/dev/video{device}')
        if not self.path_webcam.exists():
            logging.error(f"Cant find webcam device at /dev/video{device}.")
            sys.exit(-1)

        self.logger = logging.getLogger("Webcam")

        self.logger.info(f"Initializing Webcam: {width}x{height} @ {fps} Hz")
        self._video_capture = cv2.VideoCapture(device)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self._video_capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._video_capture.set(cv2.CAP_PROP_FPS, fps)
        self.logger.info(f"Webcam is set to: {self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {self._video_capture.get(cv2.CAP_PROP_FPS)} Hz")
        for _ in range(3):
            ret, frame = self._video_capture.read()
            if not ret:
                self.logger.error("Can't read webcam. Exiting.")
                sys.exit(1)
            self.frame = frame

    def read(self) -> Optional[np.ndarray]:
        """Read a single frame from webcam"""
        ret, frame = self._video_capture.read()

        if not ret:
            self.close()
            return None
        return frame

    def shape(self) -> Tuple[int, int]:
        """Read width and height of images from webcam

        Returns:
            Tuple[int, int]: Tuple of (width, height)
        """
        width = self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return (int(width), int(height))

    def close(self) -> None:
        """Release VideoCapture object"""
        self._video_capture.release()
        self.logger.info("Webcam closed.")
