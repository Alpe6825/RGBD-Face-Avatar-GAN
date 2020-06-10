#Last edit 07.06.2020
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import cv2
import Utils.HeatmapDrawing as hd
from time import sleep

class FacialLandmarkController():

    def __init__(self):

        self.matching = False;

        self.root = tk.Tk()
        self.root.title("Facial Landmark Controller")
        self.root.protocol('WM_DELETE_WINDOW', self.__CancelCommand)
        streamFrame = tk.Frame(self.root)
        streamFrame.pack()

        tk.Label(streamFrame, text="Landmark").grid(row=0, column=0)
        tk.Label(streamFrame, text="Avatar Landmarks").grid(row=0, column=1)

        self.canvas_0 = tk.Canvas(streamFrame, width=256, height=256)
        self.canvas_0.grid(row=1, column=0)
        self.canvas_1 = tk.Canvas(streamFrame, width=256, height=256)
        self.canvas_1.grid(row=1, column=1)
        self.slider_1 = tk.Scale(streamFrame, from_=0, to=0, length=256, command=self._slider)
        self.slider_1.grid(row=1, column=2)

        #self.canvas_2 = tk.Canvas(streamFrame, width=256, height=256)
        #self.canvas_2.grid(row=1, column=3)

        noise = np.random.randint(255, size=(256, 256))
        self.initial_tkImage = ImageTk.PhotoImage(Image.fromarray(noise))

        self.stream_0 = self.canvas_0.create_image(0, 0, anchor="nw", image=self.initial_tkImage)
        self.stream_1 = self.canvas_1.create_image(0, 0, anchor="nw", image=self.initial_tkImage)
        #self.stream_2 = self.canvas_2.create_image(0, 0, anchor="nw", image=self.initial_tkImage)

        self.matchButton = tk.Button(streamFrame, text="Start Matching", command=self._matchButton, state="disabled")
        self.matchButton.grid(row=2, column=0)

        self.loadButton = tk.Button(streamFrame, text="Load Profil", command=self._loadButton)
        self.loadButton.grid(row=2, column=1)

        self.loadedProfil = Image.new('RGB', (256, 256), (0, 0, 0))

        self.neutralProfil1 = None
        self.neutralProfil2 = None


    def _matchButton(self):
        if self.matchButton["text"] == "Start Matching":
            self.matchButton["text"] = "Stop Matching"
            self.matching = True;
            self.updateCanvas()
        else:
            self.matchButton["text"] = "Start Matching"
            self.matching = False;
            self.updateCanvas(1, hd.drawHeatmap(self.neutralProfil2,256,True)[:,:,0:3])

    def _loadButton(self):
        #name = filedialog.askopenfilename()
        #print(name)

        self.loadedLandmarks = np.loadtxt('Dataset/Landmarks.txt', dtype=float).reshape((-1, 2))

        self.loadedProfil = Image.new('RGB', (256, 256), (0, 0, 0))
        draw = ImageDraw.Draw(self.loadedProfil)
        for i in range(self.loadedLandmarks.shape[0]):
            x = self.loadedLandmarks[i][0]
            y = self.loadedLandmarks[i][1]

            if x > 3 and y > 3:
                draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=(55, 55, 55))

        self.loadedLandmarks = self.loadedLandmarks.reshape((-1, 70,2))
        self.slider_1.config(to=self.loadedLandmarks.shape[0]-1)
        self.slider_1.set(0)
        self._slider(0)

        self.matchButton.config(state="normal")

    def _slider(self, value):
        self.neutralProfil2 = self.loadedLandmarks[int(value)]
        self.updateCanvas(1,hd.drawHeatmap(self.neutralProfil2,256,True)[:,:,0:3])

    def updateCanvas(self,canvas=0, img=np.ones((256,256))):

        if canvas == 0:
            global tkImage_0
            pilImage = Image.fromarray(img)
            tkImage_0 = ImageTk.PhotoImage(image=pilImage)
            self.canvas_0.itemconfigure(self.stream_0, image=tkImage_0)

        if canvas == 1:
            global tkImage_1
            pilImage = Image.fromarray(img)
            pilImage = Image.composite(pilImage,self.loadedProfil,pilImage.convert("L"))
            tkImage_1 = ImageTk.PhotoImage(image=pilImage)
            self.canvas_1.itemconfigure(self.stream_1, image=tkImage_1)

        self.root.update()

    def adjustLandmarks(self, landmarks):
        dif = landmarks-self.neutralProfil1
        return self.neutralProfil2 + dif

    def __call__(self,landmarks):
        self.updateCanvas(0,hd.drawHeatmap(landmarks,256,True)[:,:,0:3])

        if self.matching == False:
            self.neutralProfil1 = landmarks
        else:
            landmarks = self.adjustLandmarks(landmarks)
            self.updateCanvas(1,hd.drawHeatmap(landmarks,256,True)[:,:,0:3])

        return landmarks

    def __CancelCommand(event=None):
        pass

if __name__ == '__main__':

    temp = np.array([7, 55, 7, 87, 7, 118, 7, 145, 15, 176, 30, 203,
              50, 223, 73, 242, 112, 254, 155, 242, 186, 223, 209, 207,
              229, 180, 237, 145, 244, 118, 248, 87, 248, 52, 26, 28,
              38, 17, 54, 13, 73, 13, 85, 17, 147, 9, 163, 5,
              182, 1, 202, 5, 213, 20, 116, 48, 116, 67, 112, 87,
              112, 106, 93, 118, 100, 122, 116, 122, 128, 122, 135, 118,
              50, 52, 61, 48, 73, 48, 89, 52, 73, 55, 61, 55,
              151, 52, 163, 44, 178, 44, 190, 48, 178, 55, 163, 55,
              65, 161, 81, 149, 104, 145, 116, 145, 128, 141, 151, 145,
              174, 149, 151, 176, 132, 188, 116, 192, 97, 192, 81, 180,
              65, 161, 100, 153, 116, 153, 132, 153, 170, 149, 132, 172,
              116, 176, 100, 176, 72, 53, 173, 53, ]).reshape(70, 2)

    flc = FacialLandmarkController()

    while True:
        landmarks = flc(temp)
        sleep(0.1)

    flc.root.mainloop()

