import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import cv2

import Utils.FaceAlignmentNetwork as fan
import Utils.EyeTracking as et
import Utils.HeatmapDrawing as hd
import Utils.CropAndResize as car
import Utils.Visualization as vis

class RGBDFaceDataset(Dataset):

    def __init__(self, path="", imageSize=256):

        self.path_rgb8 = path + "RGB/"  # Pfad zu Ordner
        self.path_depth16 = path + "Depth/"

        self.rgb8_files = [i for i in os.listdir(self.path_rgb8) if
                           i.endswith('.png') or i.endswith('.jpg')]  # Liste alle Datein auf
        self.depth16_files = [i for i in os.listdir(self.path_depth16) if i.endswith('.png')]
        self.rgb8_files.sort(key=lambda f: os.path.splitext(f)[0])  # Sortiere nach Namen
        self.depth16_files.sort(key=lambda f: os.path.splitext(f)[0])

        if len(self.rgb8_files) != len(self.depth16_files):  # Prüfen ob gleich viele RGB unf Tiefenbilder vorhanden sind
            print("Different number of rgb8- and depth16-files!")
            print("RGB: ", len(self.rgb8_files), " Depth: ", len(self.depth16_files))
            exit()

        self.imageSize = imageSize

        ### Depth Histrogramm skalieren
        max = 0
        min = 65535
        print("Compute DepthScale:")
        for file in tqdm(self.depth16_files):
            imageDepth16 = o3d.io.read_image(self.path_depth16 + file)
            tempMax = np.asarray(imageDepth16).max()
            tempMin = np.asarray(imageDepth16).min()
            if max < tempMax:
                max = tempMax
            if min > tempMin:
                min = tempMin

        self.depthScale = 65535 / max
        print("DepthRange in Dataset:", min, max, "Depthscale for 16bit:", self.depthScale)

        self.landmarks = np.ndarray([len(self.rgb8_files), 68 + 2, 2])

        if os.path.exists(path + 'Landmarks.txt'):                    # LoadLandmarks
            print('Load Landmarks')
            test = np.loadtxt(path + 'Landmarks.txt', dtype=float)
            self.landmarks = test.reshape((-1, 68 + 2, 2))
        else:
            print("Create Landmarks:")
            for idx in tqdm(range(0, len(self.rgb8_files))):
                image = o3d.io.read_image(self.path_rgb8 + self.rgb8_files[idx])  # öffne Bilder
                image = np.asarray(image).astype(float)  # Convert to Numpy Array
                image = cv2.flip(image, 0)  # Spieglen -> Bug Claymore/AzureKinect)

                landmarks = fan.create2DLandmarks(torch.Tensor(image[:, :, 0:3]))
                image, landmarks = car.cropAndResizeImageLandmarkBased(image, self.imageSize, landmarks)
                landmarks = np.concatenate((landmarks, et.eyeTracking(image[:, :, 0:3].astype("uint8"))), axis=0)

                self.landmarks[idx] = landmarks

            np.savetxt(path + 'Landmarks.txt', self.landmarks.reshape(-1))
            print("Landmarks saved as " + path + "/Landmarks.txt")

    def __len__(self):
        return len(self.rgb8_files)

    def __getitem__(self, idx):
        imageRGB8 = o3d.io.read_image(self.path_rgb8 + self.rgb8_files[idx])  # öffne Bilder
        imageDepth16 = o3d.io.read_image(self.path_depth16 + self.depth16_files[idx])

        imageRGB8 = np.asarray(imageRGB8).astype(float)  # Convert to Numpy Array
        imageDepth16 = np.asarray(imageDepth16).astype(float)

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

        fourChannelHeatmap = hd.drawHeatmap(self.landmarks[idx], self.imageSize)
        fourChannelHeatmap = (fourChannelHeatmap - 127.5) / 127.5


        sample = {'RGBD': imageRGBD, 'Heatmap': fourChannelHeatmap}

        return sample


if __name__ == '__main__':

    rgbdFaceDataset = RGBDFaceDataset(imageSize=256, path="Dataset/")

    for i in range(0, len(rgbdFaceDataset)):
        sample = rgbdFaceDataset[i]
        vis.exportExample(sample['RGBD'], sample['Heatmap'], "Dataset/Visualization/" + str(i) + ".png")
