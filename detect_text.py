import cv2
from object_detection import non_max_suppression
import numpy as np
import global_vars as gv

class DetectText:

    def __init__(self, model_path):
        self.rw = 320
        self.rh = 320

        self.net = cv2.dnn.readNet(model_path)
        self.layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        self.min_confidence = 0.5

    def detect_text(self, img):

        # img preprocess, it helps to improve accuracy!
        image = img
        (imgh, imgw) = image.shape[:2]
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        rW = imgw / float(self.rw)
        rH = imgh / float(self.rh)
        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (self.rw, self.rh))
        blob = cv2.dnn.blobFromImage(image, 1.0, (self.rw, self.rh), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layerNames)

        (rows, cols) = scores.shape[2:4]
        rects = []
        confidences = []

        # ignore the rect which score less then min-confidence
        for i in range(rows):
            scoresData = scores[0, 0, i]
            xData0 = geometry[0, 0, i]
            xData1 = geometry[0, 1, i]
            xData2 = geometry[0, 2, i]
            xData3 = geometry[0, 3, i]
            anglesData = geometry[0, 4, i]
            for j in range(cols):
                if scoresData[j] < self.min_confidence:
                    continue

                (offsetX, offsetY) = (j * 4.0, i * 4.0)
                angle = anglesData[j]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[j] + xData2[j]
                w = xData1[j] + xData3[j]

                endX = int(offsetX + (cos * xData1[j]) + (sin * xData2[j]))
                endY = int(offsetY - (sin * xData1[j]) + (cos * xData2[j]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[j])

        # execute nms
        rects = np.array(rects)
        pick = non_max_suppression(rects, probs=confidences)
        boxes = rects[pick]

        # draw the bounding boxes
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * rW).astype(np.int32)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * rH).astype(np.int32)

        return boxes
        # for i in boxes.shape[0]:
        #     boxes[i, 0] = int(boxes[i, 0] * rW)
        #     startY = int(boxes[i, 1] * rH)
        #     endX = int(boxes[i, 2] * rW)
        #     endY = int(boxes[i, 3] * rH)

    def visualize(self, img, boxes):
        for (x, y, ex, ey) in boxes:
            cv2.rectangle(img, (x, y), (ex, ey), gv.green, 2)

        return img
