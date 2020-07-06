# Last edit 06.07.2020
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import open3d as o3d
import cv2 as cv
import threading
import time
from queue import Queue
import configFile as config

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

def evalVis(input,heatmap,color,depth, export = True):
    """
    f = plt.figure(figsize=(16, 6))
    f.tight_layout()

    f1 = f.add_subplot(141)
    f2 = f.add_subplot(142)
    f3 = f.add_subplot(143)
    f4 = f.add_subplot(144)

    f1.axis('off')
    f1.imshow(input)
    f1.set_title("Camera")

    f2.imshow(heatmap)
    f2.set_title("Landmarks")
    f2.axis("off")

    f3.imshow(color)
    f3.set_title("Output (RGB)")
    f3.axis("off")

    f4.imshow(depth)
    f4.set_title("Output (Depth)")
    f4.axis("off")
    plt.show()
    """

    if export == True:
        input = cv.cvtColor(input, cv.COLOR_RGB2BGR)
        cv.imwrite("Data/" + config.DatasetName + "/Result/Snaps/camera.png",input)
        cv.imwrite("Data/" + config.DatasetName + "/Result/Snaps/heatmap.png", heatmap)
        color = cv.cvtColor(color, cv.COLOR_RGB2BGR)
        cv.imwrite("Data/" + config.DatasetName + "/Result/Snaps/outputColor.png", color)
        cv.imwrite("Data/" + config.DatasetName + "/Result/Snaps/outputDepth.png", depth)


def showPointCloud(x,depthScale = 1000, depth_trunc=1000, export=True):
    image = x.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    height, width, color = image.shape
    rgb = np.zeros((height, width, color-1), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.uint16)

    for h in range(0, height):
        for w in range(0, width):
            rgb[h][w][0] = image[h][w][0] * 127 + 127
            rgb[h][w][1] = image[h][w][1] * 127 + 127
            rgb[h][w][2] = image[h][w][2] * 127 + 127
            depth[h][w] = (image[h][w][3] * 32767 + 32767)

    mask = depth/65535*255
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv.erode(mask, kernel, iterations=1)
    depth = (depth * (mask2 / 255)).astype(np.uint16)

    """
    hist = cv.calcHist([depth], [0], None, [65536], [0, 65536])
    hist[0] = 0
    min_i = 0
    while hist[min_i, 0] == 0:
        min_i += 1
    max_i = 65535
    while hist[max_i, 0] == 0:
        max_i -= 1
    print(min_i,max_i)

    hist = cv.calcHist([depth], [0], None, [max_i-min_i], [min_i, max_i])
    plt.plot(hist)
    plt.show()

    depth = depth.reshape(-1)
    depth -= min_i
    print(depth.shape)
    depth = (depth.astype(float)/(max_i-min_i)*255*0.5).astype(np.uint8).reshape(256,256)
    """

    # depth = cv.GaussianBlur(depth, (3, 3), 0)

    color_raw = o3d.geometry.Image(rgb)
    depth_raw = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=depthScale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

    azure = o3d.camera.PinholeCameraIntrinsic()
    azure.set_intrinsics(height=1080, width=1920,
                         fx=916.9168701171875, fy=916.5850830078125,
                         cx=150, cy=200)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, azure)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if export == True:
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud("Result/Snaps/PointCloud.pts", pcd)
    else:
        return pcd

class realtimePointCloud():

    def __init__(self):

        self.queue = Queue()
        self.t = threading.Thread(target=self.test, args=("Thread-1", self.queue))
        self.t.start()

    def __call__(self, x):
        #self.geometry.translate(np.array([[0],[0.01],[0]]))
        #self._vis.update_geometry(self.geometry)
        self.queue.put(x)

    def test(self, threadname, q):
        _vis = o3d.visualization.Visualizer()
        _vis.create_window()
        geometry = o3d.geometry.TriangleMesh.create_box()
        _vis.add_geometry(geometry)

        while True:
            time.sleep(0.1)
            x = q.get()
            if x is not None:
                #return  # Poison pill

                _vis.remove_geometry(geometry)
                geometry = showPointCloud(x, export=False)
                _vis.add_geometry(geometry)
                #geometry.translate(np.array([[0],[0.01],[0]]))
                #_vis.update_geometry(geometry)

            _vis.poll_events()
            _vis.update_renderer()

