#last edit: 10.06.2020
from PIL import Image, ImageDraw
import torch
import numpy as np

def drawHeatmap(landmarks, imageSize,returnType="Numpy"):
    # make a blank image for the text, initialized to transparent text color
    image = Image.new('RGBA', (imageSize, imageSize), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    for i in range(1, landmarks.shape[0]):

        if i == 17 or i == 22 or i == 27 or i == 31:# or i == 36 or i == 42 or i == 48 or i == 60:
            continue

        prev = i - 1
        if i == 36:
            prev = 41
        elif i == 42:
            prev = 47
        elif i == 48:
            prev = 59
        elif i == 60:
            prev = 67

        x1 = landmarks[prev][0]
        y1 = landmarks[prev][1]
        x2 = landmarks[i][0]
        y2 = landmarks[i][1]

        if i == 68 or i == 69:
            if x2 > 3 and y2 > 3:
                draw.ellipse([(x2-3,y2-3),(x2+3,y2+3)],fill=(255,255,255,255))
        else:
            draw.line([x1, y1, x2, y2], width=4)


    if returnType == "PIL":
        return image
    else:
        image = np.asarray(image)
        if returnType == "Tensor":
            return torch.Tensor(image.transpose(2, 0, 1))
        else:
            return image