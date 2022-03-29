"""Module to blur background within an image using mediapipe"""
import logging
from typing import Optional

import mediapipe
import numpy as np
import cv2


class BackgroundBlur:
    def __init__(self, bg_image: Optional[np.ndarray] = None):
        """
        Initialize background blur module
        """
        self.logger = logging.getLogger("BGBlur")

        self.selfie_segmentation = mediapipe.solutions.selfie_segmentation
        self.bg_image = bg_image

    def background_blur(self, frame: np.ndarray) -> np.ndarray:
        """Detect person in image, blur background, and return modified image.
        Using Google mediapipe selfie segmentation module for this task

        Args:
            frame (np.ndarray): Image that shall be modified

        Returns:
            np.ndarray: Image with blurred background

        Reference:
            https://google.github.io/mediapipe/solutions/selfie_segmentation.html
        """
        with self.selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = selfie_segmentation.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack(
                (results.segmentation_mask,) * 3, axis=-1) > 0.1
            # The background can be customized.
            #   a) Load an image (with the same width and height of the input image) to
            #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
            #   b) Blur the input image by applying image filtering, e.g.,
            #      bg_image = cv2.GaussianBlur(image,(55,55),0)
            if self.bg_image is None:
                bg_image = cv2.GaussianBlur(image, (55, 55), 0)
            output_image = np.where(condition, image, bg_image)
        return output_image
