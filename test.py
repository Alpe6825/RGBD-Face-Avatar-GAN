# Last edit 06.07.2020
import torch
import torch.nn as nn
import Pix2PixGAN.Generator as pix2pixG
import Pix2PixGAN.Initialization as pix2pixInit
import functools
import Utils.FaceAlignmentNetwork as fan
import Utils.CropAndResize as car
import Utils.EyeTracking as et
import Utils.Visualization as vis
from Utils.FacialLandmarkControl import FacialLandmarkController
import cv2
import numpy as np
import Utils.HeatmapDrawing as hd
import pandas as pd
import configFile as config

if __name__ == '__main__':

    imageSize = 256
    flc = FacialLandmarkController()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Define Networks ###

    netG = pix2pixG.UnetGenerator(input_nc=4, output_nc=4, num_downs=8, ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=False)
    netG = pix2pixInit.init_net(netG)

    netG.load_state_dict(torch.load("Data/" + config.DatasetName + "/Result/trainedGenerator.pth"))
    netG.eval().to(device)

    print("LoadModel")
    loaded = torch.jit.load("Data/" + config.DatasetName + "/Result/tracedGenerator.zip")
    loaded = loaded.to(device)
    # print(loaded)
    # print(loaded.code)

    camID = 0
    cap = cv2.VideoCapture(camID)  # 0 for webcam or path to video
    while (cap.isOpened() == False):
        print("Error opening video stream or file")
        camID +=1
        cap = cv2.VideoCapture(camID)  # 0 for webcam or path to video
    print("Camera ID:",camID)

    #live3DPlot = vis.realtimePointCloud()

    while (True):
        # Capture frame-by-frame
        _, frame = cap.read()
        """
        datasetRGB = sorted(os.listdir("Dataset/RGB/"))
        frame = cv2.imread("Dataset/RGB/" + datasetRGB[332])
        frame = cv2.flip(frame,0)
        """

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            landmarks_temp = fan.create2DLandmarks(torch.Tensor(frame))
            #imageRGBD = car.cropAndResizeImageLandmarkBased(imageRGBD, 256, landmarks_temp, computeLandmarksAgain=False)
            image, landmarks = car.cropAndResizeImageLandmarkBased(frame, imageSize, landmarks_temp, useCropBuffer=False)
            landmarks = np.concatenate((landmarks, et.eyeTracking(image[:, :, 0:3].astype("uint8"))), axis=0)

            landmarks = flc(landmarks)

            fourChannelHeatmap = hd.drawHeatmap(landmarks, imageSize,returnType="Tensor")
            fourChannelHeatmap = (fourChannelHeatmap - 127.5) / 127.5

            outputTensor = loaded.forward(fourChannelHeatmap.unsqueeze(0).to(device))# netG(fourChannelHeatmap.unsqueeze(0).to(device)) #loaded.forward(fourChannelHeatmap.unsqueeze(0)) #
            depthScale = 64.06158357771261
            outputTensor[0,3,:,:] *= (depthScale/65535*2*255)
            output = outputTensor[0].cpu().clone().detach().numpy()

            compl = np.zeros([3, imageSize, int(imageSize * 4)])
            compl[:, :, 0:imageSize] = (image.transpose(2, 0, 1) - 127.5)/127.5
            compl[:, :, imageSize:imageSize*2] = fourChannelHeatmap[0:3, :, :]# + lanmarkCrontrolVis.transpose(2,0,1)
            compl[:, :, imageSize*2:imageSize*3] = output[0:3, :, :]
            compl[0, :, imageSize*3:imageSize*4] = output[3, :, :]
            compl[1, :, imageSize*3:imageSize*4] = output[3, :, :]
            compl[2, :, imageSize*3:imageSize*4] = output[3, :, :]

            output = compl.transpose(1, 2, 0)
            output = output * 127.5 + 127.5
            output = output.astype('uint8')

            #live3DPlot(outputTensor[0])

            if cv2.waitKey(1) & 0xFF == ord('s'):
                hm = fourChannelHeatmap[0:3].cpu().clone().detach().numpy()
                hm = hm * 127.5 + 127.5
                hm = hm.astype('uint8')

                vis.evalVis(image, hm.transpose(1,2,0), output[:, imageSize*2:imageSize*3,:], output[:, imageSize*3:imageSize*4,:])
                vis.showPointCloud(outputTensor[0])

        except Exception as e:
            print(e)
            output = frame

        # Display the resulting frame
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imshow('Window', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()