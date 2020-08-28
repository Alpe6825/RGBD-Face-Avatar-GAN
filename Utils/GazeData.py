# Last edit 03.07.2020 - Not Use (29.08.20)
import numpy as np
import os

class GazeData():

    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=str)
        print("Loaded GazeData", self.data.shape)

        listA = []
        listB = []

        for row in self.data:
            listA.append(os.path.basename(row[4]))
            listB.append((float(row[1].split(":")[1]), float(row[2])))

        self.gaze = dict(zip(listA, listB))

    def __call__(self, imagename):

        id = imagename.split("_colorImage")[0]
        #return 255 - np.array([self.gaze[id], self.gaze[id]])*255
        return (1 - np.array([self.gaze[id], self.gaze[id]])) * np.array([20, -10]) - np.array([10, -5])


