import global_vars as gv
import DigitsModel as DM
import tensorflow as tf
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


class DigitsOcr:

    def __init__(self, model_path):
        num_classes = 10
        self.img_size = 32
        min_area = 20 ** 2
        max_area = 50 ** 2

        self.min_confidence = 0.2
        # redefine the network
        self.x = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        self.y = tf.placeholder(tf.int32, [None])
        self.is_trainning = tf.placeholder(tf.bool, [])

        self.model = DM.DigitsModel(num_classes, 0)
        self.output = self.model.net(self.x, self.is_trainning)
        metric_dict = self.model.get_metrics(self.output, self.y)
        self.logits = metric_dict['logits']
        self.predictions = metric_dict['predictions']

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.saver.restore(self.sess, model_path)

        self.mser = cv2.MSER_create(_min_area=min_area, _max_area=max_area)

    def ocr_img(self, img):
        # the img is a whole number img
        # split the img into single number
        # img should be grayscale
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # bimg = img.copy()
            # _, bimg = cv2.threshold(bimg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        _, boxes = self.mser.detectRegions(img)

        flag = True
        rect = []
        for box in boxes:
            x, y, w, h = box

            # filter the unfit ratio
            if w/h > 1:
                continue

            rect.append((x, y, x+w, y+h))
            tmp = img[y:y+h, x:x+w]
            tmp = cv2.resize(tmp, (self.img_size, self.img_size))
            tmp = np.expand_dims(tmp, axis=(0, 3))

            if flag:
                images = tmp
                flag = False
            else:
                images = np.concatenate((images, tmp), axis=0)

        # no box is fit
        if flag:
            raise IndexError

        rect = np.array(rect)

        y_ = np.zeros(images.shape[0])
        logits, pred = self.sess.run([self.logits, self.predictions],
                                     feed_dict={self.x:images, self.y:y_, self.is_trainning:False})
        # try:
        confidence = logits[np.arange(pred.shape[0]), pred]
        # except IndexError:
        #     print('preditions is empty')
        #     raise Exception('PredError')


        # filter the confidence
        fil_indices = []
        for i in range(confidence.shape[0]):
            if confidence[i] > self.min_confidence:
                fil_indices.append(i)

        # remain bbox, confidence, pred
        rrect = rect[fil_indices, ...]
        rconfidence = confidence[fil_indices]
        rpred = pred[fil_indices]

        # nms
        pick = non_max_suppression(np.array(rrect), probs=rconfidence)

        rects = rrect[pick]
        preds = rpred[pick]

        return rects, preds


    def getDigit(self, rects, preds, visualize=False, img=None):

        x = []
        # if visualize:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i in range(preds.shape[0]):
            if visualize:
                cv2.rectangle(img, tuple(rects[i, 0:2]), tuple(rects[i, 2:4]), gv.green, 2)
            x.append(rects[i, 0])

        idx = np.argsort(np.array(x))

        strnum = ''
        for i in range(idx.shape[0]):
            strnum += str(preds[idx[i]])

        if visualize:
            cv2.putText(img, strnum, (x[idx[0]], rects[idx[0], 1]),
                        gv.font, 0.5, gv.blue, 2)

        return int(strnum), img





