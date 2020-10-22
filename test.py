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
import Utils.Evaluation as ev
import configFile as config
import keyboard
from threading import Thread
from queue import Queue
import onnx

def thread_osc(threadname, _queue):
    print("Start:" + threadname)

    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer

    def handler(address, *args):
        #print("ödjfhsödkjfh")
        if not _queue.empty():
            _queue.get()
        _queue.put(args)

    #def default_handler(address, *args):
    #    print(f"DEFAULT {address}: {args}")

    dispatcher = Dispatcher()
    #dispatcher.map("/faciallandmarks/eyebrows", eyebrown_handler)
    dispatcher.map("/landmarks", handler)
    #dispatcher.set_default_handler(default_handler)

    server = BlockingOSCUDPServer(("192.168.178.52", 9000), dispatcher)
    server.serve_forever()  # Blocks forever

if __name__ == '__main__':

    #_queue = Queue()
    #thread1 = Thread(target=thread_osc, args=("Thread-OSC", _queue))
    #thread1.start()


    imageSize = 256
    flc = FacialLandmarkController()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Define Networks ###

    netG = pix2pixG.UnetGenerator(input_nc=3, output_nc=4, num_downs=8, ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=False)
    netG = pix2pixInit.init_net(netG)

    netG.load_state_dict(torch.load("Data/" + config.DatasetName + "/Result/trainedGenerator_epoch_30.pth"))
    netG.eval().to(device)

    print("LoadModel")

    noise = torch.randn(1, 3, 256, 256)
    traced = torch.jit.trace(netG.cpu().eval(), noise)
    netG.train().to(device)
    traced.save("Data/" + config.DatasetName + "/Result/tracedGenerator_epoch_30.zip")
    exit()

    camID = 0
    cap = cv2.VideoCapture(camID)  # 0 for webcam or path to video
    while (cap.isOpened() == False):
        print("Error opening video stream or file")
        camID +=1
        cap = cv2.VideoCapture(camID)
    print("Camera ID:", camID)

    ssim_counter = 0
    ssim_modulo = 25 # Frameinterval für structural_similarity
    diff = np.zeros((imageSize, imageSize))

    #live3DPlot = vis.realtimePointCloud()

    while (True):
        # Capture frame-by-frame
        _, frame = cap.read()
        """
        datasetRGB = sorted(os.listdir("Dataset/RGB/"))
        frame = cv2.imread("Dataset/RGB/" + datasetRGB[332])
        frame = cv2.flip(frame,0)
        """
        #frame = cv2.imread("C:/Users/Alexander Pech/Pictures/Camera Roll/NME/raw7.jpg")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            # FAN
            #landmarks_temp = fan.create2DLandmarks(torch.Tensor(frame))
            #image, landmarks = car.cropAndResizeImageLandmarkBased(frame, imageSize, landmarks_temp, useCropBuffer=False)
            #landmarks = np.concatenate((landmarks, et.eyeTracking(image[:, :, 0:3].astype("uint8"))), axis=0)

            image = cv2.imread(
                "D:/Alexander_Pech/Python-Projects/RGBD-Face-Avatar-GAN/Data/Philipp-Known-Setting-RGBDFaceAvatarGAN-Datensatz03-09-2020/Color/color_and_depth_shoot_from_2020-9-2---23-46-52_colorImage_1.png",
                cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            crop_region = np.loadtxt('D:/Alexander_Pech/Python-Projects/RGBD-Face-Avatar-GAN/Data/Philipp-Known-Setting-RGBDFaceAvatarGAN-Datensatz03-09-2020/crop_region.txt', dtype=float)
            image = car.cropAndResizeImageDatasetBased(image, 256, crop_region)

            # OSC
            #landmarks = _queue.get()
            #landmarks = np.array(landmarks).reshape(70, 2)
            #print(landmarks)

            #landmarks = flc(landmarks)

            #fourChannelHeatmap = hd.drawHeatmap(landmarks, imageSize,returnType="Tensor")
            fourChannelHeatmap = torch.Tensor(image.transpose(2, 0, 1))
            fourChannelHeatmap = (fourChannelHeatmap - 127.5) / 127.5
            #print(fourChannelHeatmap.shape)

            # print(fourChannelHeatmap[:,0].unsqueeze(0).unsqueeze(0).shape) [0].unsqueeze(0)

            outputTensor = netG(fourChannelHeatmap.unsqueeze(0).to(device)) #loaded.forward(fourChannelHeatmap.unsqueeze(0)) #
            #outputTensor = loaded_1.forward(fourChannelHeatmap[0].unsqueeze(0).unsqueeze(0).to(device))  #

            depthScale = 64.06158357771261
            outputTensor[0,3,:,:] *= (depthScale/65535*2*255)
            output = outputTensor[0].cpu().clone().detach().numpy()

            compl = np.zeros([3, imageSize, int(imageSize * 5)])
            compl[:, :, 0:imageSize] = (image.transpose(2, 0, 1) - 127.5)/127.5
            compl[:, :, imageSize:imageSize*2] = fourChannelHeatmap[0:3, :, :]# + lanmarkCrontrolVis.transpose(2,0,1)
            compl[:, :, imageSize*2:imageSize*3] = output[0:3, :, :]
            compl[0, :, imageSize*3:imageSize*4] = output[3, :, :]
            compl[1, :, imageSize*3:imageSize*4] = output[3, :, :]
            compl[2, :, imageSize*3:imageSize*4] = output[3, :, :]

            if ssim_counter % ssim_modulo == 0:
                _, diff = ev.ssim(image, output[0:3, :, :].transpose(1, 2, 0))
            ssim_counter += 1
            compl[0, :, imageSize * 4:imageSize * 5] = diff/255 * 2 - 1
            compl[1, :, imageSize * 4:imageSize * 5] = diff/255 * 2 - 1
            compl[2, :, imageSize * 4:imageSize * 5] = diff/255 * 2 - 1

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