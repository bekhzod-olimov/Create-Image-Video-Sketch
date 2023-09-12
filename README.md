# Video2Sketch

This repository contains on converting video to sketch videos. It has three main steps:
1. Obtain frames from the input video;
2. Convert the frames to sketches;
3. Create video using the sketch frames.

### * Obtain frames from the input video
The video frames can be obtained using the [get_video_frames function](https://github.com/bekhzod-olimov/Create-Image-Video-Sketch/blob/e8730aaaec2d50665fb49a766194f8cc3e6ff9d7/utils.py#L14C5-L34C56) of the [VideoMaker class](https://github.com/bekhzod-olimov/Create-Image-Video-Sketch/blob/899c258091f561b82820e3650b27fab3d2a961ce/utils.py).

### * Convert the frames to sketches
After obtaining the video frames from the step 1, the images can be converted to sketches. This process can be done using two methods:

- a simple ML method: faster, lower quality;
- a more powerful DL method*: slower, higher quality.
  
As the DL methods, we use two different models. The first one is a pretrained semantic segmentation model called [U2Net semantic segmentation model](https://github.com/xuebinqin/U-2-Net).
And the other one is a pretrained GAN model called [Facial Colorization](https://github.com/SystemErrorWang/FacialCartoonization).
