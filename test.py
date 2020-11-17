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

    imageSize = config.IMAGE_SIZE
    flc = FacialLandmarkController()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Define Networks ###

    netG = pix2pixG.UnetGenerator(input_nc=config.INPUT_CHANNEL, output_nc=4, num_downs=8, ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=False)
    netG = pix2pixInit.init_net(netG)

    netG.load_state_dict(torch.load("Data/" + config.DatasetName + "/Result/trainedGenerator.pth"))
    netG.eval().to(device)
    print("LoadModel")

    """noise = torch.randn(1, 1, 256, 256)
    traced = torch.jit.trace(netG.cpu().eval(), noise)
    netG.train().to(device)
    traced.save("Data/" + config.DatasetName + "/Result/tracedGenerator_epoch_" + str(e) + ".zip")
    exit()
    """

    cap = cv2.VideoCapture(config.TEST_VIDEO)  # 0 for webcam or path to video
    camID = 0
    while (cap.isOpened() == False):
        print("Error opening video stream or file")
        cap = cv2.VideoCapture(camID)
        camID += 1
    print("Camera ID:", camID-1)

    if config.TEST_INPUT == "OSC":
        _queue = Queue()
        thread1 = Thread(target=thread_osc, args=("Thread-OSC", _queue))
        thread1.start()

    ssim_counter = 0
    ssim_modulo = 25 # Frameinterval für structural_similarity
    diff = np.zeros((imageSize, imageSize))

    #live3DPlot = vis.realtimePointCloud()

    while (True):
        # Capture frame-by-frame
        _, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            # FAN
            if config.TEST_INPUT == "Camera":
                frame = car.cropAndResizeImageDatasetBased(frame, imageSize, np.array(
                    [480, 480 + 248, 260, 260 +282]))
                landmarks_temp = fan.create2DLandmarks(torch.Tensor(frame))
                image, landmarks = frame, landmarks_temp # car.cropAndResizeImageLandmarkBased(frame, imageSize, landmarks_temp, useCropBuffer=False)
                landmarks = np.concatenate((landmarks, et.eyeTracking(image[:, :, 0:3].astype("uint8"))), axis=0)

            # OSC
            if config.TEST_INPUT == "OSC":
                landmarks = _queue.get()
                landmarks = np.array(landmarks).reshape(70, 2)
                print(landmarks)

            if config.USE_FLC:
                landmarks = flc(landmarks)

            fourChannelHeatmap = hd.drawHeatmap(landmarks, imageSize, returnType="Tensor")
            fourChannelHeatmap = (fourChannelHeatmap - 127.5) / 127.5

            outputTensor = netG(fourChannelHeatmap[0].unsqueeze(0).unsqueeze(0).to(device)) #loaded.forward(fourChannelHeatmap.unsqueeze(0)) #
            #outputTensor = loaded_1.forward(fourChannelHeatmap[0].unsqueeze(0).unsqueeze(0).to(device))  #

            output = outputTensor[0].cpu().clone().detach().numpy()

            compl = np.zeros([3, imageSize, int(imageSize * 5)])
            compl[:, :, 0:imageSize] = (image.transpose(2, 0, 1) - 127.5)/127.5
            compl[:, :, imageSize:imageSize*2] = fourChannelHeatmap[0:3, :, :]
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