import torch
import torch.nn as nn
import Pix2PixGAN.Generator as pix2pixG
import Pix2PixGAN.Initialization as pix2pixInit
import functools
import Dataset.RGBDFaceDataset as rgbdDataset
import Utils.Visualization as Vis
import Utils.FaceAlignmentNetwork as fan
import Utils.CropAndResize as car
import Utils.EyeTracking as et
import cv2
import numpy as np
import Utils.HeatmapDrawing as hd

if __name__ == '__main__':

    imageSize = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Define Networks ###

    netG = pix2pixG.UnetGenerator(input_nc=4, output_nc=4, num_downs=8, ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=True)
    netG = pix2pixInit.init_net(netG)

    netG.load_state_dict(torch.load("Result/trainedGenerator.pth"))
    netG.eval().to(device)

    cap = cv2.VideoCapture(0) # 0 for webcam or path to video

    while (True):
        # Capture frame-by-frame
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks_temp = fan.create2DLandmarks(torch.Tensor(frame))
        #imageRGBD = car.cropAndResizeImageLandmarkBased(imageRGBD, 256, landmarks_temp, computeLandmarksAgain=False)
        image, landmarks = car.cropAndResizeImageLandmarkBased(frame, imageSize, landmarks_temp)
        landmarks = np.concatenate((landmarks, et.eyeTracking(image[:, :, 0:3].astype("uint8"))), axis=0)

        fourChannelHeatmap = hd.drawHeatmap(landmarks, imageSize)
        fourChannelHeatmap = (fourChannelHeatmap - 127.5) / 127.5

        output = netG(torch.Tensor(fourChannelHeatmap.unsqueeze(0)).to(device))
        output = output[0].cpu().clone().detach().numpy()

        compl = np.zeros([3, imageSize, int(imageSize * 4)])
        compl[:, :, 0:imageSize] = (image.transpose(2, 0, 1) - 127.5)/127.5
        compl[:, :, imageSize:imageSize*2] = fourChannelHeatmap[0:3, :, :]
        compl[:, :, imageSize*2:imageSize*3] = output[0:3, :, :]
        compl[0, :, imageSize*3:imageSize*4] = output[3, :, :]
        compl[1, :, imageSize*3:imageSize*4] = output[3, :, :]
        compl[2, :, imageSize*3:imageSize*4] = output[3, :, :]

        output = compl.transpose(1, 2, 0)
        output = output * 127.5 + 127.5
        output = output.astype('uint8')

        # Display the resulting frame
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imshow('Window', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()