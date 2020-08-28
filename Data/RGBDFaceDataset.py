# Last edit 06.07.2020
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm
from PIL import Image
import cv2

import configFile as config
import Utils.FaceAlignmentNetwork as fan
import Utils.EyeTracking as et
import Utils.GazeData as gd
import Utils.HeatmapDrawing as hd
import Utils.CropAndResize as car
import Utils.Visualization as vis
import Utils.IR_EyeTracking as ir

class RGBDFaceDataset(Dataset):

    def __init__(self, path="", imageSize=256):

        self.path_rgb8 = path + "Color/"  # Pfad zu Ordner
        self.path_depth16 = path + "Depth/"
        self.path_ir8 = path + "IR/"

        self.rgb8_files = [i for i in os.listdir(self.path_rgb8) if
                           i.endswith('.png') or i.endswith('.jpg')]  # Liste alle Datein auf
        self.depth16_files = [i for i in os.listdir(self.path_depth16) if i.endswith('.png')]
        self.ir8_files = [i for i in os.listdir(self.path_ir8) if i.endswith('.png')]

        self.rgb8_files.sort(key=lambda f: os.path.splitext(f)[0])  # Sortiere nach Namen
        self.depth16_files.sort(key=lambda f: os.path.splitext(f)[0])
        self.ir8_files.sort(key=lambda f: os.path.splitext(f)[0])

        if len(self.rgb8_files) != len(self.depth16_files) and len(self.rgb8_files) != len(self.ir8_files):  # Prüfen ob gleich viele RGB unf Tiefenbilder vorhanden sind
            print("Different number of rgb8- and depth16-files!")
            print("RGB: ", len(self.rgb8_files), " Depth: ", len(self.depth16_files), " IR: ", len(self.ir8_files))
            exit()

        self.imageSize = imageSize
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.RandomAffine((-5,5), translate=(0.02,0.02), scale=(0.98,1), shear=None, resample=False,
                                            fillcolor=0),torchvision.transforms.ToTensor()])

        ### Depth Histrogramm skalieren
        if os.path.exists(path + 'Depthscale.txt'):
            print('Load Depthscale')
            self.depthScale = np.loadtxt(path + 'Depthscale.txt', dtype=float)
            print("Depthscale:", self.depthScale)
        else:
            _max = 0
            _min = 65535
            print("Compute DepthScale:")
            for file in tqdm(self.depth16_files):
                imageDepth16 = o3d.io.read_image(self.path_depth16 + file)
                tempMax = np.asarray(imageDepth16).max()
                tempMin = np.asarray(imageDepth16).min()
                if _max < tempMax:
                    _max = tempMax
                if _min > tempMin:
                    _min = tempMin

            self.depthScale = 65535 / _max
            print("DepthRange in Dataset:", _min, _max, "Depthscale for 16bit:", self.depthScale)

        self.landmarks = np.ndarray([len(self.rgb8_files), 68 + 2, 2])

        if os.path.exists(path + 'Landmarks.txt'):                    # LoadLandmarks
            print('Load Landmarks')
            test = np.loadtxt(path + 'Landmarks.txt', dtype=float)
            self.landmarks = test.reshape((-1, 68 + 2, 2))
        else:
            print("Create Landmarks:")
            #if os.path.exists(path + "GazeData.txt"):
            #    self.gaze = gd.GazeData(path + "GazeData.txt")

            eye_tracking_ir = ir.IREyeTraking(540, 320, 160, 40)

            for idx in tqdm(range(0, len(self.rgb8_files))):
                image = o3d.io.read_image(self.path_rgb8 + self.rgb8_files[idx])  # öffne Bilder
                image = np.asarray(image).astype(float)  # Convert to Numpy Array
                ir_image = cv2.imread(self.path_ir8 + self.ir8_files[idx], cv2.IMREAD_GRAYSCALE)

                if config.FlipYAxis:
                    image = cv2.flip(image, 0)  # Spieglen -> Bug Claymore/AzureKinect)
                    ir_image = cv2.flip(ir_image, 0)

                landmarks = fan.create2DLandmarks(torch.Tensor(image[:, :, 0:3]))

                x1, y1, x2, y2 = eye_tracking_ir(ir_image)
                eyesTensor = torch.Tensor([[x1, y1], [x2, y2]])
                landmarks = torch.cat((landmarks, eyesTensor), 0)

                image, landmarks = car.cropAndResizeImageLandmarkBased(image, self.imageSize, landmarks)
                self.landmarks[idx]  = landmarks

                """lefteye = landmarks[36:42, :]
                lefteye = np.mean(lefteye.numpy(), axis=0).reshape((1,2))

                righteye = landmarks[42:48, :]
                righteye = np.mean(righteye.numpy(), axis=0).reshape((1,2))

                eyekeypoints = np.concatenate((lefteye, righteye), axis=0)
                """


                #if self.gaze:
                #    eyekeypoints = eyekeypoints + self.gaze(self.rgb8_files[idx])
                """else:
                    eyekeypoints = et.eyeTracking(image[:, :, 0:3].astype("uint8"))
                """

                #self.landmarks[idx] = np.concatenate((landmarks, eyekeypoints), axis=0)

            np.savetxt(path + 'Landmarks.txt', self.landmarks.reshape(-1))
            print("Landmarks saved as " + path + "Landmarks.txt")

            """landmarkControl = np.ndarray([68 + 2, 4])

            for i in range(0, 70):
                landmarkControl[i, 0] = min(self.landmarks[:, i, 0])
                landmarkControl[i, 2] = max(self.landmarks[:, i, 0])
                landmarkControl[i, 1] = min(self.landmarks[:, i, 1])
                landmarkControl[i, 3] = max(self.landmarks[:, i, 1])

            pd.DataFrame(landmarkControl, columns=["x_min", "y_min", "x_max", "y_max"]).to_csv(
                path + 'LandmarkControl.csv')
            """

    def __len__(self):
        return len(self.rgb8_files)

    def __getitem__(self, idx):
        imageRGB8 = o3d.io.read_image(self.path_rgb8 + self.rgb8_files[idx])
        imageDepth16 = o3d.io.read_image(self.path_depth16 + self.depth16_files[idx])

        imageRGB8 = np.asarray(imageRGB8).astype(float)  # Convert to Numpy Array
        imageDepth16 = np.asarray(imageDepth16).astype(float)

        if config.FlipYAxis:
            imageRGB8 = cv2.flip(imageRGB8, 0)  # Spiegeln da Bilder von Kinect falschrum sind
            imageDepth16 = cv2.flip(imageDepth16, 0)

        if imageRGB8.shape[0:2] != imageDepth16.shape:  # Prüfen ob RGB und D zusammenpassen
            print("RGB8 und Depth16 sind unterschiedlich groß.")
            exit()

        imageRGBD = np.ndarray([imageRGB8.shape[0], imageRGB8.shape[1], 4])  # ,dtype=float
        imageRGBD[:, :, 0:3] = imageRGB8
        imageRGBD[:, :, 3] = imageDepth16 * self.depthScale

        """ Visualisierung für Histogramnormalisierung (vorher/nachher)
        plt.hist(imageDepth16.flatten()/65535, color='b', histtype='bar', range=(0, 1), bins=500, log=True)
        plt.ylabel('Anzahl der Pixel')
        plt.xlabel('normierter Tiefenwert')
        plt.show()
        cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE);
        cv2.imshow("Depth", imageDepth16/65535);
        cv2.waitKey(0);
        plt.hist(imageRGBD[:,:,3].flatten()/65535, color='b', histtype='bar', range=(0, 1), bins=500, log=True)
        plt.ylabel('Anzahl der Pixel')
        plt.xlabel('normierter Tiefenwert')
        plt.show()
        cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE);
        cv2.imshow("Depth", imageRGBD[:,:,3]/65535);
        cv2.waitKey(0);
        exit()
        """

        landmarks_temp = fan.create2DLandmarks(torch.Tensor(imageRGBD[:, :, 0:3]))
        imageRGBD = car.cropAndResizeImageLandmarkBased(imageRGBD, self.imageSize, landmarks_temp,
                                                         computeLandmarksAgain=False)
        imageRGBD[:, :, 0:3] = (imageRGBD[:, :, 0:3] - 127.5) / 127.5
        imageRGBD[:, :, 3] = (imageRGBD[:, :, 3] - 32767.5) / 32767.5

        minThresh = imageRGBD[:, :, 3].min()
        for h in range(imageRGBD.shape[0]):
            for w in range(imageRGBD.shape[1]):
                if imageRGBD[h, w, 3] == minThresh:
                    imageRGBD[h, w, 3] = 1  # -1

        imageRGBD = imageRGBD.transpose(2, 0, 1)
        imageRGBD = torch.Tensor(imageRGBD)

        fourChannelHeatmap_PIL = hd.drawHeatmap(self.landmarks[idx], self.imageSize,returnType="PIL")

        fourChannelHeatmap = self.transforms(fourChannelHeatmap_PIL)
        fourChannelHeatmap = fourChannelHeatmap*2-1
        #fourChannelHeatmap = (fourChannelHeatmap - 127.5) / 127.5

        sample = {'RGBD': imageRGBD, 'Heatmap': fourChannelHeatmap}
        return sample


if __name__ == '__main__':

    rgbdFaceDataset = RGBDFaceDataset(imageSize=256, path="Data/" + config.DatasetName + "/")

    if not os.path.exists("Data/" + config.DatasetName + "/Visualization/"):
        os.mkdir("Data/" + config.DatasetName + "/Visualization/")

    print("Visualization:")
    for i in tqdm(range(0, len(rgbdFaceDataset))):
        sample = rgbdFaceDataset[i]
        vis.exportExample(sample['RGBD'], sample['Heatmap'], "Data/" + config.DatasetName + "/Visualization/" + str(i) + ".png")
