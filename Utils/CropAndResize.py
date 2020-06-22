# Last edit 19.06.2020
import cv2
import torch
import Utils.FaceAlignmentNetwork as fan

min_x_buffer = [0, 0, 0]
max_x_buffer = [100, 100, 100]
min_y_buffer = [0, 0, 0]
max_y_buffer = [100, 100, 100]

def cropAndResizeImageLandmarkBased( image, imageSize, landmarks, computeLandmarksAgain=True, useCropBuffer=False):
    min_x = abs(landmarks[:, 0].min())
    max_x = abs(landmarks[:, 0].max())
    min_y = abs(landmarks[:, 1].min())
    max_y = abs(landmarks[:, 1].max())

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    if delta_x > delta_y:
        min_y = min_y - (delta_x - delta_y) / 2
        max_y = max_y + (delta_x - delta_y) / 2
    if delta_y > delta_x:
        min_x = min_x - (delta_y - delta_x) / 2
        max_x = max_x + (delta_y - delta_x) / 2

    if min_x < 0 or min_y < 0 or max_y > image.shape[0] - 1 or max_x > image.shape[1] - 1:

        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > image.shape[1] - 1:
            max_x = image.shape[1] - 1
        if max_y > image.shape[0] - 1:
            max_y = image.shape[0] - 1
        print("Gesicht zu nah am Bildrand!")

    if useCropBuffer == True:
        del min_x_buffer[0]
        del max_x_buffer[0]
        del min_y_buffer[0]
        del max_y_buffer[0]

        min_x_buffer.append(min_x)
        max_x_buffer.append(max_x)
        min_y_buffer.append(min_y)
        max_y_buffer.append(max_y)

        min_x = torch.Tensor([sum(min_x_buffer)/3])
        max_x = torch.Tensor([sum(max_x_buffer)/3])
        min_y = torch.Tensor([sum(min_y_buffer)/3])
        max_y = torch.Tensor([sum(max_y_buffer)/3])

    # frame = cv2.rectangle(frame, (int(min_x), int(min_y)),  (int(max_x), int(max_y)), (0, 255, 0), 5)
    frame = image[int(min_y):int(max_y), int(min_x):int(max_x)]
    frame = cv2.resize(frame, (imageSize, imageSize))

    if computeLandmarksAgain == True:
        #preds2 = fan.create2DLandmarks(torch.Tensor(frame[:, :, 0:3]))
        preds2 = landmarks
        preds2[:, 0] -= min_x
        preds2[:, 1] -= min_y

        preds2[:, 0] *= 256 / (max_x - min_x)
        preds2[:, 1] *= 256 / (max_y - min_y)
        return frame, preds2
    else:
        return frame
