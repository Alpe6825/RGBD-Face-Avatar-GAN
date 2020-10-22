import cv2
from skimage.metrics import structural_similarity #as ssim
import numpy
import time
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import Utils.CropAndResize as car
import torch
import Utils.FaceAlignmentNetwork as fan
import Utils.IR_EyeTracking as ir
import Utils.HeatmapDrawing as hd
import os

def ssim(img1, img2, range8bit=False):

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #img1 = img1 * 127.5 + 127.5
    #img1 = img1.astype('uint8')
    if range8bit == False:
        img2 = img2 * 127.5 + 127.5
        img2 = img2.astype('uint8')

    start = time.time()
    (score, diff) = structural_similarity(img1, img2, full=True)
    print("Image similarity:", score, "(", time.time()-start, "s)")

    return score, diff * 255


def diff(img1, img2):

    #print(img1.shape, img2.shape)

    diff = cv2.absdiff(img1, img2)

    #plt.imshow(diff)
    #plt.show()
    #diff = cv2.equalizeHist(diff)
    return diff


def nothing(x) -> None:
    pass

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

if __name__ == "__main__":
    mk_path = "C:/Users/Alexander Pech/Desktop/SortedErodesV2/neuesNeutralBild/"

    path_GAN = mk_path + "gan" + ".png"
    path_IR = mk_path + "ir"  + ".png"
    path_rgb = mk_path + "rgb"  + ".png"
    path_depth = mk_path + "depth"  + ".png"
    depthScale = 63.25772200772201

    cv2.namedWindow("image")
    cv2.createTrackbar('X', 'image', 25, 50, nothing)
    cv2.createTrackbar('Y', 'image', 25, 50, nothing)
    cv2.createTrackbar('W', 'image', 25, 50, nothing)
    cv2.createTrackbar('H', 'image', 25, 50, nothing)
    cv2.createTrackbar('R', 'image', 25, 50, nothing)

    ### Pre GAN Pipline
    start = time.time()
    while True:

        imageDepth16 = o3d.io.read_image(path_depth)
        imageDepth16 = np.asarray(imageDepth16).astype(float)
        imageDepth16 = rotate_image(imageDepth16, cv2.getTrackbarPos("R", "image")-25)

        crop_region = np.array([451 + cv2.getTrackbarPos("X", "image") - 25,
                                699 + cv2.getTrackbarPos("W", "image") - 25,
                                286 + cv2.getTrackbarPos("Y", "image") - 25,
                                568 + cv2.getTrackbarPos("H", "image") - 25])
        imageDepth16 = car.cropAndResizeImageDatasetBased(imageDepth16, 256, crop_region)

        imageDepth16 *= depthScale
        imageDepthFloat = (imageDepth16 - 32767.5) / 32767.5

        #print(imageDepthFloat.min(), imageDepthFloat.max())

        ### Erode cpp
        imageDepth8 = (imageDepthFloat * 127.5 + 127.5).astype("uint8")
        """"t = (imageDepth8.max() - imageDepth8.min())/4 + imageDepth8.min()

        _, mask = cv2.threshold(imageDepth8, int(t), 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((5,5),np.uint8), iterations=1)

        imageDepth8 = imageDepth8 & mask"""
        #cv2.imshow("mask", imageDepth8)
        #cv2.waitKey(100000)
        #exit()
        hist = cv2.calcHist([imageDepth8], [0], None, [75], [75, 150])
        plt.plot(hist)

        ### DIff

        gan = cv2.imread(path_GAN, cv2.IMREAD_UNCHANGED)
        absdiff = diff(imageDepth8, gan[:, :, 3])
        hist2 = cv2.calcHist([gan[:, :, 3]], [0], None, [75], [75, 150])
        plt.plot(hist2)
        plt.show()

        ### FAN

        imageRGB8 = cv2.imread(path_rgb, cv2.IMREAD_UNCHANGED)
        imageRGB8 = rotate_image(imageRGB8, cv2.getTrackbarPos("R", "image")-25)


        landmarks = fan.create2DLandmarks(torch.Tensor(imageRGB8[:, :, 0:3]), show=False)
        eye_tracking_ir = ir.IREyeTraking(500, 330, 160, 40)
        x1, y1, x2, y2 = eye_tracking_ir(cv2.imread(path_IR, cv2.IMREAD_GRAYSCALE))
        eyesTensor = torch.Tensor([[x1, y1], [x2, y2]])
        #print(eyesTensor)
        landmarks = torch.cat((landmarks, eyesTensor), 0)

        landmarks = car.cropAndResizeLandmarksDatasetBased(landmarks, 256, crop_region)

        fourChannelHeatmap = hd.drawHeatmap(landmarks, 256, returnType="numpy")

        imageRGB8 = car.cropAndResizeImageDatasetBased(imageRGB8, 256, crop_region)

        ### SSIM
        score, ssim_image = ssim(imageRGB8, gan[:, :, 0:3], range8bit=True)

        _image = np.concatenate((ssim_image/1, absdiff/1, absdiff*50/1), axis=1).astype("uint8")
        _image = cv2.cvtColor(_image, cv2.COLOR_GRAY2BGR)
        _image = np.concatenate((_image, imageRGB8[:, :, 0:3], gan[:, :, 0:3]), axis=1)

        #print(time.time() - start)
        start = time.time()
        cv2.imshow("image", _image)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    ### Safe
    mk_path += "eval/"
    os.mkdir(mk_path)

    cv2.imwrite(mk_path + "DerGeraet_RGB.png", imageRGB8)
    cv2.imwrite(mk_path + "DerGeraet_Depth.png", imageDepth8)

    cv2.imwrite(mk_path + "GAN_RGB.png", gan[:, :, 0:3])
    cv2.imwrite(mk_path + "GAN_Depth.png", gan[:, :, 3])

    #cv2.imwrite(mk_path + "Diff_Depth.png", absdiff)

    cv2.imwrite(mk_path + "SSIM_RGB.png", ssim_image)
    np.savetxt(mk_path + "ssim_score.txt", np.array([score]))

    cv2.imwrite(mk_path + "FAN.png", fourChannelHeatmap[:, :, 0])

    plt.imshow(absdiff)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(mk_path + "Diff_Depth.png", format="png" )

    while True:
        pass
    """

    color_raw = o3d.io.read_image("C:/Users/Alexander Pech/Desktop/SortedErodes/New folder - Neutral sehr gut/eval/GAN_RGB.png")
    depth_raw = o3d.io.read_image("C:/Users/Alexander Pech/Desktop/SortedErodes/New folder - Neutral sehr gut/eval/GAN_Depth.png")

    #depth_raw = (np.asarray(depth_raw)).astype("float")
    #_min = depth_raw.min()
    #_max = depth_raw.max()

    depth_raw = o3d.geometry.Image(depth_raw)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000, depth_trunc=1000, convert_rgb_to_intensity=False)

    azure = o3d.camera.PinholeCameraIntrinsic()
    azure.set_intrinsics(height=1080, width=1920,
                         fx=916.9168701171875, fy=916.5850830078125,
                         cx=150, cy=200)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, azure)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd])
    """

