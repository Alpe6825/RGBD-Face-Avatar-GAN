# RGBD-Face-Avatar-GAN

This repository based on my bachelorthesis at HS-Düsseldorf and was supported by [MIREVI](www.mirevi.de). 

The basic GAN architecture is the [Pix2Pix-GAN by Isola et al.](https://phillipi.github.io/pix2pix/). 
Some basic code parts came from or are based on their [Pix2Pix-Pytorch-implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

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

You need rgbd-images in form of an 8bit rgb image and 16bit depth image. 
Put the rgb images under "Dataset/RGB/" and the depth images under "Dataset/Depth/".

A tool for creating the dataset automatical with an rgbd-camera is still under construction.

If you run `RGBDfaceDataset.py`, folder "Visualization" contains the complete dataset in a visible form. 
Ths is not necessary for training and only for debug.

## Training

Run `train.py`.