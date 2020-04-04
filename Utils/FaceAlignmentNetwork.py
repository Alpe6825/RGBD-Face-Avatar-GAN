import face_alignment
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def create2DLandmarks(imageTensor, show=False):

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cpu")

    input = imageTensor.detach().numpy()

    if input.shape[0] < input.shape[2]:
        input = input.transpose(1, 2, 0)

    with torch.no_grad():  ## SUPER Wichtig zerstört sonst den Autograd prozess
        preds = fa.get_landmarks(input[:, :, 0:3])

    if preds == None:
        preds = np.zeros((1, 68, 2))

    if show == True:
        h, w, c = input.shape

        landmarks_preview = np.zeros((h, w), dtype="float")

        for lm in preds[0]:
            x = int(lm[1])
            y = int(lm[0])
            landmarks_preview[x - 4:x + 4, y - 4:y + 4] = 1

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Landmarks')
        ax1.imshow(input / 255)
        ax2.imshow(landmarks_preview)
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    return torch.Tensor(preds[0])


if __name__ == "__main__":
    #input = io.imread('testbild2.png')
    input = cv2.imread('testImage.png', cv2.IMREAD_UNCHANGED)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    testTensor = torch.Tensor(input)
    landmarks = create2DLandmarks(testTensor, show=True)
    print(landmarks.shape)