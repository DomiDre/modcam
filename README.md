# ModCam

Reads the camera via opencv2 (by default set to MJPEG with 1920x1080 setting),
then uses mediapipes "Selfie Segmentation" to blur background,
and finally uses v4l2loopback and ffmpeg to write the modified image to a virtual camera,
which should be useable in a video call like a normal webcam.

## Requirements

Install v4l2loopback

``` bash
git clone git@github.com:umlaeute/v4l2loopback.git
cd v4l2loopback
make
sudo make install
sudo depmod -a
```

## Install Modcam

``` bash
pip3 install -r requirements.txt
```

### Usage

``` bash
python3 modcam.py
```

Use --help for optional arguments