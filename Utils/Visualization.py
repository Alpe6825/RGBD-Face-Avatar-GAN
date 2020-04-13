import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def showDatapair(image, heatmaps):
    image = image.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    h, w, c = image.shape
    image.reshape(-1)
    image = image * 0.5 + 0.5
    image.reshape([h, w, c])
    # print("Shape:", image.shape, "RGB:", np.min(image[:, :, 0:3]), np.max(image[:, :, 0:3]), "Depth:", np.min(image[:, :, 3]), np.max(image[:, :, 3]))

    fig = plt.figure()
    fig.add_subplot(221).imshow(image[:, :, 0:3])
    fig.add_subplot(222).imshow(image[:, :, 3])

    heatmaps = heatmaps.cpu().clone().detach().numpy()
    heatmaps = heatmaps.transpose(1, 2, 0)
    h, w, c = heatmaps.shape
    heatmaps.reshape(-1)
    heatmaps = heatmaps * 0.5 + 0.5
    heatmaps.reshape([h, w, c])
    fig.add_subplot(223).imshow(heatmaps[:, :, 0])
    # fig.add_subplot(224).imshow(fA.sumOfHeatmaps(heatmaps))

    plt.show()
    #plt.show(block=False)
    #plt.pause(2)
    #plt.close()

def exportExample(image, heatmaps, path):

    image = image.cpu().clone().detach().numpy()
    heatmaps = heatmaps.cpu().clone().detach().numpy()

    compl = np.zeros([3, 256, 512+256])
    compl[:, :, 0:256] = heatmaps[0:3, :, :]
    compl[:, :, 256:512] = image[0:3, :, :]
    compl[0, :, 512:768] = image[3, :, :]
    compl[1, :, 512:768] = image[3, :, :]
    compl[2, :, 512:768] = image[3, :, :]

    compl = compl.transpose(1, 2, 0)
    compl = compl * 127.5 + 127.5
    compl = compl.astype('uint8')

    im = Image.fromarray(compl)
    im.save(path)


