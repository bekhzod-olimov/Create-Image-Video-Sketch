# Video2Sketch

This repository contains on converting video to sketch videos. It has three main steps:
1. Obtain frames from the input video;
2. Convert the frames to sketches;
3. Create video using the sketch frames.

### * Obtain frames from the input video
The video frames can be obtained using the [get_video_frames function](https://github.com/bekhzod-olimov/Create-Image-Video-Sketch/blob/e8730aaaec2d50665fb49a766194f8cc3e6ff9d7/utils.py#L14C5-L34C56) of the [VideoMaker class](https://github.com/bekhzod-olimov/Create-Image-Video-Sketch/blob/899c258091f561b82820e3650b27fab3d2a961ce/utils.py).
##### Frame samples:

![frame10](https://github.com/bekhzod-olimov/Create-Image-Video-Sketch/assets/50166164/c6ce1f18-3a5e-4cc2-99d6-45e354a15016)
![frame133](https://github.com/bekhzod-olimov/Create-Image-Video-Sketch/assets/50166164/2d3d0592-3a25-43d2-aeea-b19ea185328d)
![frame120](https://github.com/bekhzod-olimov/Create-Image-Video-Sketch/assets/50166164/1e3bc0a7-edbc-4737-944a-dfa5e5b22115)


### * Convert the frames to sketches
After obtaining the video frames from the step 1, the images can be converted to sketches. This process can be done using two methods:

- a simple ML method: faster, lower quality;
- a more powerful DL method*: slower, higher quality.
  
As the DL models, we use two different networks. The first one is a pretrained semantic segmentation model called [U2Net](https://github.com/xuebinqin/U-2-Net).
And the other one is a pretrained GAN model called [Facial Colorization](https://github.com/SystemErrorWang/FacialCartoonization).

##### Sketch samples using U2Net:

![frame10](https://github.com/bekhzod-olimov/Create-Image-Video-Sketch/assets/50166164/143cfba4-7d0a-44ce-9880-756bc5f015ea)
![frame133](https://github.com/bekhzod-olimov/Create-Image-Video-Sketch/assets/50166164/c28d4764-de9a-47de-b11d-63aa9eb4ccbd)
![frame120](https://github.com/bekhzod-olimov/Create-Image-Video-Sketch/assets/50166164/b33d0939-932f-429e-9fa8-ec0f98a92ac5)
