import torch
import torch.nn as nn
import Pix2PixGAN.Generator as pix2pixG
import Pix2PixGAN.Initialization as pix2pixInit
import functools
import Dataset.RGBDFaceDataset as rgbdDataset
import Utils.FaceAlignmentNetwork as fan
import Utils.CropAndResize as car
import Utils.EyeTracking as et
import Utils.Visualization as vis
import cv2
import numpy as np
import Utils.HeatmapDrawing as hd
import onnx
import pandas as pd
import os

if __name__ == '__main__':

    imageSize = 256
    landmarkControl = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Define Networks ###

    netG = pix2pixG.UnetGenerator(input_nc=4, output_nc=4, num_downs=8, ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=True)
    netG = pix2pixInit.init_net(netG)

    ### ONNX ####

    # Input to the model
    x = torch.randn(1, 4, 256, 256, requires_grad=True)

    # Export the model
    torch.onnx.export(netG,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "Result/tracedGenerator.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})
    onnx_model = onnx.load("Result/tracedGenerator.onnx")
    print(onnx.checker.check_model(onnx_model))

    ##############

    netG.load_state_dict(torch.load("Result/trainedGenerator.pth"))
    netG.eval().to(device)

    camID = 0
    cap = cv2.VideoCapture(camID)  # 0 for webcam or path to video
    while (cap.isOpened() == False):
        print("Error opening video stream or file")
        camID +=1
        cap = cv2.VideoCapture(camID)  # 0 for webcam or path to video
    print("Camera ID:",camID)

    df = pd.read_csv("Result/LandmarkControl.csv")
    lanmarkCrontrolVis = np.ndarray([imageSize,imageSize,3])

    if landmarkControl == True:
        for i in range(0,70):
            cv2.rectangle(lanmarkCrontrolVis, (int(df['x_min'][i]), int(df['y_min'][i])), (int(df['x_max'][i]), int(df['y_max'][i])), (100, 0, 0), 2)


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
            image, landmarks = car.cropAndResizeImageLandmarkBased(frame, imageSize, landmarks_temp)
            landmarks = np.concatenate((landmarks, et.eyeTracking(image[:, :, 0:3].astype("uint8"))), axis=0)

            if landmarkControl == True:
                for i in range(0,70):
                    if landmarks[i,0] < df['x_min'][i]:
                        landmarks[i,0] = df['x_min'][i]
                    elif landmarks[i,0] > df['x_max'][i]:
                        landmarks[i,0] = df['x_max'][i]
                    if landmarks[i,1] < df['y_min'][i]:
                        landmarks[i,1] = df['y_min'][i]
                    elif landmarks[i,1] > df['y_max'][i]:
                        landmarks[i,1] = df['y_max'][i]


            fourChannelHeatmap = hd.drawHeatmap(landmarks, imageSize)
            fourChannelHeatmap = (fourChannelHeatmap - 127.5) / 127.5

            outputTensor = netG(torch.Tensor(fourChannelHeatmap.unsqueeze(0)).to(device))
            output = outputTensor[0].cpu().clone().detach().numpy()

            compl = np.zeros([3, imageSize, int(imageSize * 4)])
            compl[:, :, 0:imageSize] = (image.transpose(2, 0, 1) - 127.5)/127.5
            compl[:, :, imageSize:imageSize*2] = fourChannelHeatmap[0:3, :, :] + lanmarkCrontrolVis.transpose(2,0,1)
            compl[:, :, imageSize*2:imageSize*3] = output[0:3, :, :]
            compl[0, :, imageSize*3:imageSize*4] = output[3, :, :]
            compl[1, :, imageSize*3:imageSize*4] = output[3, :, :]
            compl[2, :, imageSize*3:imageSize*4] = output[3, :, :]

            output = compl.transpose(1, 2, 0)
            output = output * 127.5 + 127.5
            output = output.astype('uint8')

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