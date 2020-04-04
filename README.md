# RGBD-Face-Avatar-GAN

This repository based on my bachelorthesis at HS-Düsseldorf and is supported by MIREVI. 

### Requirements

- Python 3.6 or 3.7 
- numpy
- PyTorch 1.4 (with Cuda 10.1)
- Torchvision 0.5
- Open3D
- OpenCV 4
- matplotlib
- tqdm
- face-alignment (install PyTorch before)



## Create Dataset

You need rgbd-images in form an 8bit rgb image and 16bit depth image. 
Put the rgb images under "Dataset/RGB/" and the depth images under "Dataset/Depth/".

An tool for creating the dataset automatical with an rgbd-camera is still under construction.