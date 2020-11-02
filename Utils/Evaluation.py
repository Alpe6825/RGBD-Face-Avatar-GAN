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
import Pix2PixGAN.Generator as pix2pixG
import Pix2PixGAN.Initialization as pix2pixInit
import functools
import configFile as config
import torch.nn as nn
import Utils.Visualization as vis

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


def apply_depthmask(img):
    ret, mask = cv2.threshold(img, 250, 1, cv2.THRESH_BINARY_INV)
    #plt.imshow(mask)
    #plt.show()
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    #plt.imshow(img)
    #plt.show()
    img = img * mask + (1 - mask) * 255
    #plt.imshow(img)
    #plt.show()
    # print("applied mask", np.min(img[np.nonzero(img)]), img.max())
    return img

if __name__ == "__main__":
    mk_path = "C:/Users/Alexander Pech/Desktop/____finale Auswahl NEU3/H/"

    #path_GAN = mk_path + "gan" + ".png"
    path_IR = mk_path + "ir"  + ".png"
    path_rgb = mk_path + "rgb"  + ".png"
    path_depth = mk_path + "depth"  + ".png"
    #depthScale = 63.25772200772201

    cv2.namedWindow("image")
    cv2.createTrackbar('X', 'image', 25, 50, nothing)
    cv2.createTrackbar('Y', 'image', 25, 50, nothing)
    cv2.createTrackbar('W', 'image', 25, 50, nothing)
    cv2.createTrackbar('H', 'image', 25, 50, nothing)
    cv2.createTrackbar('R', 'image', 25, 50, nothing)

    if os.path.exists(mk_path + "eval/"):
        poses = np.loadtxt(mk_path + "eval/TrackbarPosesForCropss.txt", dtype=int)
        cv2.setTrackbarPos('X', 'image', poses[0])
        cv2.setTrackbarPos('Y', 'image', poses[1])
        cv2.setTrackbarPos('W', 'image', poses[2])
        cv2.setTrackbarPos('H', 'image', poses[3])
        cv2.setTrackbarPos('R', 'image', poses[4])


    netG = pix2pixG.UnetGenerator(input_nc=1, output_nc=4, num_downs=8, ngf=64,
                                  norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True),
                                  use_dropout=False)
    netG = pix2pixInit.init_net(netG)
    netG.load_state_dict(torch.load("Data/" + config.DatasetName + "/Result/trainedGenerator.pth"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG.eval().to(device)

    eye_tracking_ir = ir.IREyeTraking(500, 330, 170, 30)

    ### Pre GAN Pipline
    start = time.time()
    while True:

        crop_region = np.array([451 + cv2.getTrackbarPos("X", "image") - 25,
                                699 + cv2.getTrackbarPos("W", "image") - 25,
                                286 + cv2.getTrackbarPos("Y", "image") - 25,
                                568 + cv2.getTrackbarPos("H", "image") - 25])

        debug = np.zeros((512, 1024, 3))

        ### realDepth

        imageDepth16 = o3d.io.read_image(path_depth)
        imageDepth16 = np.asarray(imageDepth16).astype(float)
        imageDepth16 = rotate_image(imageDepth16, cv2.getTrackbarPos("R", "image") - 25)
        imageDepth16 = car.cropAndResizeImageDatasetBased(imageDepth16, 256, crop_region)

        imageDepth16 = imageDepth16 - config.DEPTH_OFFSET
        for x in range(0, imageDepth16.shape[1]):
            for y in range(0, imageDepth16.shape[0]):
                temp = imageDepth16[y, x]
                if temp <= 0:
                    temp = 255
                if temp > 255:
                    temp = 255
                imageDepth16[y, x] = temp

        imageDepth8 = imageDepth16.astype("uint8")
        imageDepth8 = apply_depthmask(imageDepth8)

        debug[256:512, 256:512, 0] = imageDepth8
        debug[256:512, 256:512, 1] = imageDepth8
        debug[256:512, 256:512, 2] = imageDepth8

        ### realRGB & FAN

        imageRGB8 = cv2.imread(path_rgb, cv2.IMREAD_UNCHANGED)
        imageRGB8 = rotate_image(imageRGB8, cv2.getTrackbarPos("R", "image") - 25)

        landmarks = fan.create2DLandmarks(torch.Tensor(imageRGB8[:, :, 0:3]), show=False)

        ir_image = cv2.imread(path_IR, cv2.IMREAD_GRAYSCALE)
        ir_image = rotate_image(ir_image, cv2.getTrackbarPos("R", "image") - 25)
        x1, y1, x2, y2 = eye_tracking_ir(ir_image)
        eyesTensor = torch.Tensor([[x1, y1], [x2, y2]])
        # print(eyesTensor)
        landmarks = torch.cat((landmarks, eyesTensor), 0)
        landmarks = car.cropAndResizeLandmarksDatasetBased(landmarks, 256, crop_region)
        fourChannelHeatmap = hd.drawHeatmap(landmarks, 256, returnType="numpy")

        imageRGB8 = car.cropAndResizeImageDatasetBased(imageRGB8, 256, crop_region)

        ret, mask = cv2.threshold(imageDepth8, 110, 1, cv2.THRESH_BINARY_INV)
        mask[30:180, 65:190] = 1
        imageRGB8[:, :, 0] *= mask
        imageRGB8[:, :, 1] *= mask
        imageRGB8[:, :, 2] *= mask


        debug[0:256, 0:256, :] = fourChannelHeatmap[:, :, 0:3]
        debug[0:256, 256:512, :] = imageRGB8[:, :, 0:3]

        ### GAN
        heatmap = (fourChannelHeatmap[:, :, 1] - 127.5) / 127.5

        gan = netG(torch.Tensor(heatmap).unsqueeze(0).unsqueeze(0).to(device))
        gan = gan[0].cpu().detach().numpy().transpose((1, 2, 0))
        gan = ((gan + 1) * 127.5).astype("uint8")

        gan[:, :, 3] = apply_depthmask(gan[:, :, 3])
        gan[:, :, 3] = apply_depthmask(gan[:, :, 3])

        gan_orig = np.copy(gan)
        ret, mask = cv2.threshold(gan[:, :, 3], 110, 1, cv2.THRESH_BINARY_INV)
        gan[:, :, 0] *= mask
        gan[:, :, 1] *= mask
        gan[:, :, 2] *= mask

        debug[0:256, 512:768, :] = cv2.cvtColor(gan[:, :, 0:3], cv2.COLOR_RGB2BGR)
        debug[256:512, 512:768, 0] = gan[:, :, 3]
        debug[256:512, 512:768, 1] = gan[:, :, 3]
        debug[256:512, 512:768, 2] = gan[:, :, 3]

        ### SSIM
        score, ssim_image = ssim(imageRGB8, cv2.cvtColor(gan[:, :, 0:3], cv2.COLOR_RGB2BGR), range8bit=True)
        debug[0:256, 768:1024, 0] = ssim_image
        debug[0:256, 768:1024, 1] = ssim_image
        debug[0:256, 768:1024, 2] = ssim_image

        ### Diff Depths + Hist

        absdiff = diff(imageDepth8, gan[:, :, 3])

        debug[256:512, 768:1024, 0] = absdiff * 21
        debug[256:512, 768:1024, 1] = absdiff * 21
        debug[256:512, 768:1024, 2] = absdiff * 21

        cv2.imshow("image", debug.astype("uint8"))
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    ### Safe
    mk_path += "eval/"
    if not os.path.exists(mk_path):
        os.mkdir(mk_path)

    cv2.imwrite(mk_path + "DerGeraet_RGB.png", imageRGB8)
    cv2.imwrite(mk_path + "DerGeraet_Depth.png", imageDepth8)

    cv2.imwrite(mk_path + "GAN_RGB.png", cv2.cvtColor(gan[:, :, 0:3], cv2.COLOR_RGB2BGR))
    cv2.imwrite(mk_path + "GAN_Depth.png", gan[:, :, 3])

    cv2.imwrite(mk_path + "SSIM_RGB.png", ssim_image)
    np.savetxt(mk_path + "ssim_score.txt", np.array([score]))

    cv2.imwrite(mk_path + "FAN.png", fourChannelHeatmap[:, :, 0])

    plt.imshow(absdiff)
    plt.colorbar(label="Difference (mm)")
    plt.axis('off')
    plt.clim(0, 13)
    plt.savefig(mk_path + "Diff_Depth.png", format="png" )
    plt.close()

    hist = cv2.calcHist([imageDepth8], [0], None, [254], [0, 254])
    plt.plot(hist, label="Ground Truth")
    hist2 = cv2.calcHist([gan[:, :, 3]], [0], None, [254], [0, 254])
    plt.plot(hist2, label="Prediction")
    plt.legend(loc="upper right")
    plt.title("Depth Histogram:")
    plt.xlabel("Depth (mm)")
    plt.ylabel("Number of Pixels")
    plt.savefig(mk_path + "Depth_Histo.png", format="png")

    trackbars = np.array([cv2.getTrackbarPos("X", "image"),
                          cv2.getTrackbarPos("Y", "image"),
                          cv2.getTrackbarPos("W", "image"),
                          cv2.getTrackbarPos("H", "image"),
                          cv2.getTrackbarPos("R", "image"),
                          ])
    np.savetxt(mk_path + "TrackbarPosesForCropss.txt", trackbars)

    rgbd = np.zeros((256, 256, 4))
    rgbd[:, :, 0:3] = (gan_orig[:, :, 0:3] - 127.5) / 127.5
    rgbd[:, :, 3] = (gan_orig[:, :, 3] - 127.5) / 127.5

    # Tool: https://www.andre-gaschler.com/rotationconverter/
    trans = [[0.8663667,  0.0445444,  0.4974178, 0],
             [0.0445444,  0.9851518, -0.1658060, 0],
             [-0.4974178,  0.1658060,  0.8515186, 0],
             [0, 0, 0, 1]]

    vis.showPointCloud(rgbd,  transform=trans)

    trans = [[0.0000000, 0.0000000, -1.0000000, 0.0000000],
             [0.0000000, 1.0000000, 0.0000000, 0.0000000],
             [1.0000000, 0.0000000, 0.0000000, 0.0000000],
             [0, 0, 0, 1]]
    vis.showPointCloud(rgbd, transform=trans)



