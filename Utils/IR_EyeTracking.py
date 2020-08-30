import cv2
import numpy as np
from os import listdir


def nothing(x):
    pass

#def maprange(a, b, s):
#    (a1, a2), (b1, b2) = a, b
#    return b1 + ((s - a1) * (b2 - b1) / (a2 - a1))
"""
def detect_eyes(img, cascade=cv2.CascadeClassifier('haarcascade_eye.xml')):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = []
    left_eye_offset_x = 0
    left_eye_offset_y = 0
    right_eye = []
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
"""

class IREyeTraking:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        cv2.namedWindow("window2")
        cv2.createTrackbar("Thresh", "window2", 120, 255, nothing) #106

    def __call__(self, image) -> (int, int, int, int):
        image = cv2.equalizeHist(image)
        roi = image[self.y:self.y+self.h, self.x:self.x+self.w]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(image,(self.x, self.y), (self.x + self.w, self.y + self.h), (0, 0, 255), 2)

        roi[:, int(self.w / 3):int(self.w / 3) * 2] = 255

        roi = cv2.GaussianBlur(roi, (11, 11), 0)
        thresh = cv2.getTrackbarPos("Thresh", "window2")
        _, roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        roi = cv2.dilate(roi, kernel, iterations=2)

        #roi_left = []
        #roi_right = []
        roi_left = roi[:, 0:int(self.w / 2)]
        roi_right = roi[:, int(self.w / 2):self.w]

        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0

        contours, _ = cv2.findContours(roi_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in contours:
            (x1, y1, w1, h1) = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

            y1 += self.y + int(h1 / 2)
            x1 += self.x + w1 - 15

            image[y1 - 3:y1 + 3, x1 - 3:x1 + 3] = np.array([0, 255, 0])

            break


        contours, _ = cv2.findContours(roi_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        offset_w = int(self.w / 2)
        for cnt in contours:
            (x2, y2, w2, h2) = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x2 + offset_w, y2), (x2 + w2 + offset_w, y2 + h2), (0, 255, 0), 2)

            y2 += self.y + int(h2 / 2)
            x2 += self.x + int(self.w/2) + 15

            image[y2 - 3:y2 + 3, x2 - 3:x2 + 3] = np.array([0, 255, 0])

            break


        cv2.imshow("window2", roi)
        cv2.imshow("IR-Image", image)
        cv2.waitKey(1)

        return x1, y1, x2, y2


if __name__ == "__main__":

    cv2.namedWindow("window2")
    cv2.createTrackbar("Thresh", "window2", 106, 255, nothing)

    test = IREyeTraking(540, 320, 160, 40)

    files = [file for file in listdir("../Data/3.Durchlauf-4.Datensatz-OhneLampeImHintergrundUndOhneStuhllehne/IR/") if file.endswith('.png')]#
    while True:
        for file in files:
            image = cv2.imread("../Data/3.Durchlauf-4.Datensatz-OhneLampeImHintergrundUndOhneStuhllehne/IR/" + file, cv2.IMREAD_GRAYSCALE)
            image = cv2.flip(image, 0)

            test(image)
            continue

            ########
            """left_roi, right_roi, _, _, _, _ = detect_eyes(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR))
            if len(left_roi) == 0:
                print("no_left")
                continue
            cv2.imshow("left", left_roi)
            if len(right_roi) == 0:
                print("no_right")
                continue
            print(right_roi)
            cv2.imshow("right", right_roi)"""

            image = cv2.equalizeHist(image)
            roi = image[320:360, 540:700]
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


            #roi =  maprange( (0, 255), (200, 255), roi)
            #roi = np.array([maprange((0, 255), (200, 255), x) for x in roi])

            #roi = np.clip(roi, 0, 15)
            #roi = roi * (255/15)
            #roi = roi.astype("uint8")

            _, w = roi.shape
            roi[:, int(w / 3):int(w / 3) * 2] = 255

            roi = cv2.GaussianBlur(roi, (11, 11), 0)
            thresh = cv2.getTrackbarPos("Thresh", "window2")
            _, roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY_INV)

            #roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            kernel = np.ones((5, 5), np.uint8)
            roi = cv2.dilate(roi, kernel, iterations=2)
            #cv2.imshow("window2", roi)


            roi_left = []
            roi_right = []
            roi_left = roi[:, 0:int(w / 2)]
            roi_right = roi[:, int(w / 2):w]
            #cv2.imshow("roi_left2", roi_left)

            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

            """contours, _ = cv2.findContours(roi_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(roi_left, (x, y), (x + w, y + h), (0, 0, 255), 2)

                params = cv2.SimpleBlobDetector_Params()
                params.filterByColor = True
                params.blobColor = 0

                params.filterByCircularity = True
                params.minCircularity = 0.8
                params.maxCircularity = 1.0

                params.filterByConvexity = True
                params.minConvexity = 0.6
                params.maxConvexity = 1.0

                detector = cv2.SimpleBlobDetector_create(params)
                keypoints = detector.detect(roi_left[y:y+h,x:x+w])

                if len(keypoints) > 0:
                    print(roi_left.shape)
                    roi_left = cv2.drawKeypoints(roi_left[y:y+h,x:x+w], keypoints, np.array([]), 255,
                                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    y += 320
                    x += 540
                    image[y:y + h, x:x + w] = cv2.drawKeypoints(image[y:y + h, x:x + w], keypoints, np.array([]), (0, 0, 255),
                                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                else:
                    cv2.imshow("left_norm", image[y:y + h, x:x + w])
                    print(np.mean(roi_left))

                break"""

            contours, _ = cv2.findContours(roi_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            offset_w = int(w/2)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x + offset_w, y), (x + w + offset_w, y + h), (0, 255, 0), 2)

                y += 320 + int(h/2)
                x += 540 + 80 + 15

                image[y-3:y+3, x-3:x + 3] = np.array([0, 255, 0])

                break

            contours, _ = cv2.findContours(roi_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                y += 320 + int(h / 2)
                x += 540 + w - 15

                image[y - 3:y + 3, x - 3:x + 3] = np.array([0, 255, 0])

                break

            cv2.imshow("window2", roi)
            #cv2.imshow("roi_left", roi_left)
            #cv2.imshow("roi_right", roi_right)
            cv2.imshow("Full", image)
            cv2.waitKey(100)
            continue

            #roi = cv2.GaussianBlur(roi, (3,3), 0)
            #thresh = cv2.getTrackbarPos("Thresh", "window")
            #_, roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)




            _, w = roi.shape
            roi_left = roi[:, 0:int(w/2)]
            roi_right = roi[:, int(w/2):w]



            """
            gray_pixels = np.where(roi_left != 255)
            if gray_pixels[0] != []:
                min = np.min(gray_pixels[0])
                max = np.max(gray_pixels[0])
                left_y = int((min+max)/2)
                roi_left[left_y,:] = 0
            else:
                print("Error left eye Y")
            if gray_pixels[1] != []:
                min = np.min(gray_pixels[0])
                max = np.max(gray_pixels[0])
                left_x = int((min+max)/2)
                roi_left[:, left_x] = 0

            cv2.imshow("test_left", roi_left)"""



            thresh = cv2.getTrackbarPos("Thresh", "window")
            _, roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY_INV)
            #roi = cv2.GaussianBlur(roi, (3, 3), 0)

            #kernel = np.ones((2, 2), np.uint8)
            #roi = cv2.dilate(roi, kernel)
            #roi = cv2.erode(roi, kernel)



            contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print(len(contours))
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            for con in contours:
                area = cv2.contourArea(con)
                print(area)
                #if 15 > area > 2:
                cv2.drawContours(roi, con, -1, (0, 255, 0), 2)

                    # M = cv2.moments(con)
                    # cX = int(M["m10"] / M["m00"])
                    # cY = int(M["m01"] / M["m00"])
                    # cv2.circle(image, (cX+550, cY+330), 3, (0, 0, 255))


            """params = cv2.SimpleBlobDetector_Params()

            # Filter by Area.
            params.filterByArea = True
            params.minArea = 0

            # Create a detector with the parameters
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR))

            roi = cv2.drawKeypoints(roi, keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            """

            cv2.imshow("window", roi)
            #cv2.imshow("Norm", cv2.equalizeHist(image))
            #image[left_y-2+300:left_y+2+300, left_x-2+540:left_x+2+540] = 255
            cv2.imshow("Full", image)

            cv2.waitKey(300)
