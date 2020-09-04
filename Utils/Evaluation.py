import cv2
from skimage.metrics import structural_similarity #as ssim
import numpy
import time
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import Utils.CropAndResize as car

def ssim(img1, img2):

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    #img1 = img1 * 127.5 + 127.5
    #img1 = img1.astype('uint8')
    img2 = img2 * 127.5 + 127.5
    img2 = img2.astype('uint8')

    start = time.time()
    (score, diff) = structural_similarity(img1, img2, full=True)
    print("Image similarity:", score, "(", time.time()-start, "s)")

    return diff


def diff(img1, img2):

    print(img1.shape, img2.shape)

    diff = cv2.absdiff(img1, img2)

    plt.imshow(diff)
    plt.show()


if __name__ == "__main__":


    path = "//Claymore/c/GAN-Temp/test/color_and_depth_shoot_from_2020-9-4---3-27-41_depthImage_1.png"
    depthScale = 63.25772200772201

    ### Pre GAN Pipline

    imageDepth16 = o3d.io.read_image(path)
    imageDepth16 = np.asarray(imageDepth16).astype(float)

    crop_region = np.array([451, 699, 286, 568])
    imageDepth16 = car.cropAndResizeImageDatasetBased(imageDepth16, 256, crop_region)

    imageDepth16 *= depthScale
    imageDepthFloat = (imageDepth16 - 32767.5) / 32767.5

    print(imageDepthFloat.min(), imageDepthFloat.max())

    ### Erode cpp
    imageDepth8 = (imageDepthFloat * 127.5 + 127.5).astype("uint8")
    print(imageDepth8.min(), imageDepth8.max())

    _, mask = cv2.threshold(imageDepth8, 127, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.erode(mask, np.ones((5,5),np.uint8), iterations=1)

    imageDepth8 = imageDepth8 & mask

    ### DIff

    gan = cv2.imread("//Claymore/c/GAN-Temp/test/erodedDiffDepth2020-9-4---3-26-53-erodedDiff.png", cv2.IMREAD_UNCHANGED)[:,: ,3]
    absdiff = diff(imageDepth8, gan)

