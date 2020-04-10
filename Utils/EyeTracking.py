# Based on https://github.com/stepacool/Eye-Tracker/blob/No_GUI/track.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import Utils.FaceAlignmentNetwork as fan
import Utils.CropAndResize as car

# init part
eye_cascade = cv2.CascadeClassifier('Utils/haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = [None]
    left_eye_offset_x = 0
    left_eye_offset_y = 0
    right_eye = [None]
    right_eye_offset_x = 0
    right_eye_offset_y = 0
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            left_eye_offset_x = x
            left_eye_offset_y = y
        else:
            right_eye = img[y:y + h, x:x + w]
            right_eye_offset_x = x
            right_eye_offset_y = y
    return left_eye, right_eye, left_eye_offset_x, left_eye_offset_y, right_eye_offset_x, right_eye_offset_y


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

    return img, eyebrow_h


def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray_frame)
    #plt.show()
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    #plt.imshow(img)
    #plt.show()
    img = cv2.erode(img, None, iterations=2)
    #img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    #plt.imshow(img)
    #plt.show()
    print(keypoints)
    return keypoints


def eyeTracking(image):

    landmarks = np.zeros((2, 2))

    left_eye, right_eye, left_eye_offset_x, left_eye_offset_y, right_eye_offset_x, right_eye_offset_y = detect_eyes(image, eye_cascade)



    if len(left_eye) > 1:
        eye, offsetY = cut_eyebrows(left_eye)
        left_eye_offset_y += offsetY

        threshold = 20
        while True:
            keypoints = blob_process(eye, threshold, detector)
            keypoint = cv2.KeyPoint_convert(keypoints)
            if len(keypoint) > 0:
                landmarks[0] = keypoint[0]
                landmarks[0][0] += left_eye_offset_x
                landmarks[0][1] += left_eye_offset_y
                break
            threshold += 5;

    if len(right_eye) > 1:
        eye, offsetY = cut_eyebrows(right_eye)
        right_eye_offset_y += offsetY

        threshold = 20
        while True:
            keypoints = blob_process(eye, threshold, detector)
            keypoint = cv2.KeyPoint_convert(keypoints)
            if len(keypoint) > 0:
                landmarks[1] = keypoint[0]
                landmarks[1][0] += right_eye_offset_x
                landmarks[1][1] += right_eye_offset_y
                break
            threshold += 5;

    return landmarks


if __name__ == "__main__":

    print("Eye Tracking Test")

    cap = cv2.VideoCapture("/Utils/Testvideo.mp4")

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        testImage = frame

        landmarks = fan.create2DLandmarks(torch.Tensor(testImage))
        testImage = car.cropAndResizeImageLandmarkBased(testImage, 256, landmarks, False)

        landmarks = eyeTracking(testImage)
        for l in range(0, 2):
            x, y = landmarks[l]

            x = int(x)
            y = int(y)

            testImage[y - 2:y + 2, x - 2:x + 2, 0] = 0
            testImage[y - 2:y + 2, x - 2:x + 2, 1] = 0
            testImage[y - 2:y + 2, x - 2:x + 2, 2] = 255

        # Display the resulting frame
        cv2.imshow('frame', testImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

