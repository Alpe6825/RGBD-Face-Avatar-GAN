import cv2
from skimage.metrics import structural_similarity #as ssim
import numpy
import time

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


"""
image1 = cv2.imread('')
image2 = cv2.imread('C:/devel/RGBD-Face-Avatar-GAN/Data/3.Durchlauf-4.Datensatz-OhneLampeImHintergrundUndOhneStuhllehne/Result/10/outputColor.png')

# image1 = cv2.imread('C:/devel/RGBD-Face-Avatar-GAN/testSSIM1000.jpg') # 90% JPG C.
# image2 = cv2.imread('C:/devel/RGBD-Face-Avatar-GAN/testSSIM10000.jpg') # nur 38 JPG
# Convert images to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = ssim(image1_gray, image2_gray, full=True)
print("Image similarity:", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] image1 we can use it with OpenCV
diff = (diff * 255).astype("uint8")

cv2.imshow('diff', diff)
cv2.imwrite('C:/devel/RGBD-Face-Avatar-GAN/Data/3.Durchlauf-4.Datensatz-OhneLampeImHintergrundUndOhneStuhllehne/Result/10/diff.png', diff)

err = numpy.sum((image1_gray.astype("float") - image2_gray.astype("float")) ** 2)
err /= float(image1_gray.shape[0] * image1_gray.shape[1])
print("Image err: ", err)
# test = cv2.norm(image1_gray, image2_gray, normType=cv2.NORM_L2)
# cv2.imshow('test', err)

squares = (image1_gray[:,:3] - image2_gray[:,:3]) ** 2
numpy.sum(squares)
#cv2.imshow('test', squares)

cv2.waitKey()
"""