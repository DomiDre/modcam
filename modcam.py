import logging
from pathlib import Path
import argparse
from modules import Webcam, VirtualCam, BackgroundBlur
import signal
import sys

import cv2


logging.basicConfig(level="INFO")
parser = argparse.ArgumentParser()
parser.add_argument("--webcam", default=0, type=int, help="Number X of device in /dev/videoX which represents webcam stream")
parser.add_argument("--virtualdevice", default=2, type=int, help="Number X of device in /dev/videoX which shall be used for the virtual webcam (created by v4l2loopback)")
parser.add_argument("--bgimage", default=None, type=str, help="Path to background image to be used instead of blur")
parser.add_argument("--width", default=1920, type=int, help="Target width that is requested from the webcam")
parser.add_argument("--height", default=1080, type=int, help="Target height that is requested from the webcam")
parser.add_argument("--fps", default=30, type=int, help="Target frames/s that are requested from the webcam")
parser.add_argument("--show", default=False, action="store_true", help="Display window showing the modified stream")
args = parser.parse_args()

webcam = Webcam(args.width, args.height, args.fps, device=args.webcam)
bgblur = BackgroundBlur()
vcam = VirtualCam(video_device=args.virtualdevice)

# get camera image & set vcam to image shape
image = webcam.read()
assert image is not None, "Cant read webcam"
vcam.width = image.shape[1]
vcam.height = image.shape[0]
vcam.init_ffmpeg()

if args.bgimage is not None:
    path_bg_image = Path(args.bgimage)
    if path_bg_image.exists():
        bgblur.bg_image = cv2.imread(str(path_bg_image))
        bgblur.set_bg_image_size(vcam.width, vcam.height)
    else:
        print(f"Set bgimage, but file does not exist: {path_bg_image}")


def sigint_handler(signal, frame):
    """Catch sigints and shutdown gracefully."""
    webcam.close()
    sys.exit()


signal.signal(signal.SIGINT, sigint_handler)  # catch sigint

# run indefinitely or until q is pressed
while True:
    image = webcam.read()
    if image is None:
        print("Couldnt read webcam. Stopping")
        continue
    modified_image = bgblur.background_blur(image)
    rgb_frame = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
    vcam.write(rgb_frame)
    if args.show:
        cv2.imshow('Blurred Background Cam', modified_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
