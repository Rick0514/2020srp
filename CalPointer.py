import cv2
import os
import global_vars as gv
import numpy as np
import pickle as pk

class CalPointer:

    def __init__(self, imgdir, params_path=None):

        self.save_path = params_path
        if params_path is None or (not os.path.exists(params_path)):

            self.calPointerWinname = r'CalibratePointer'
            self.rotateImgWinname = r'RotateImg'

            self.img, self.imgtoShow = None, None
            self.imgdir = imgdir

            # 2 slope
            self.slope = [None, None]
            self.pt = [(-1,-1), (-1,-1), (-1,-1)]

            self.flagDone = 0
            self.cnt = 0

            # Bisector
            self.bisector = None
            self.len = 300

            # img rotate angle
            self.angle = None
            self.drawCursorCnt = 0
            self.drawCursorPt = []

        else:
            with open(params_path, 'rb') as f:
                data = pk.load(f)

            self.slope = data['slope']
            self.pt = data['pt']
            self.bisector = data['bisector']
            self.angle = data['angle']


    def drawPointer(self, event, x, y, flags, params):
        if self.flagDone < 2:
            if event == cv2.EVENT_LBUTTONUP:
                cv2.circle(self.imgtoShow, (x, y), 5, gv.red, 2)
                text = '({:d}, {:d})'.format(x, y)
                cv2.putText(self.imgtoShow, text, (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gv.white, 2)
                # save the pt
                self.pt[self.cnt] = (x,y)

                self.cnt += 1
                if self.cnt == 2:
                    cv2.line(self.imgtoShow, self.pt[0], self.pt[1], gv.green, 2)
                    num = self.pt[0][0] - self.pt[1][0]
                    den = self.pt[0][1] - self.pt[1][1]
                    if den == 0:
                        den = 1e-3
                    self.slope[self.flagDone] = num / den
                    self.flagDone += 1

                elif self.cnt == 3:
                    cv2.line(self.imgtoShow, self.pt[1], self.pt[2], gv.green, 2)
                    num = self.pt[1][0] - self.pt[2][0]
                    den = self.pt[1][1] - self.pt[2][1]
                    if den == 0:
                        den = 1e-3
                    self.slope[self.flagDone] = num / den
                    self.flagDone += 1


                cv2.imshow(self.calPointerWinname, self.imgtoShow)

    def drawCursortoFindAngle(self, event, x, y, flags, params):
        if self.drawCursorCnt <= 2:
            if event == cv2.EVENT_LBUTTONUP:
                self.drawCursorCnt += 1
                # save the pointer to draw line
                self.drawCursorPt.append((x, y))
                cv2.circle(self.imgtoShow, (x, y), 5, gv.blue, 2)
                if self.drawCursorCnt == 2:
                    cv2.line(self.imgtoShow, self.drawCursorPt[0], self.drawCursorPt[1], gv.green, 2)

                cv2.imshow(self.rotateImgWinname, self.imgtoShow)


    # use before calibrate-pointer
    def getRotateAngle(self):
        flagDone = False  # weather the img is chosen
        print('choose one img to find angle [y/n]')

        # when flagDone is true, the img is picked!
        for each in os.listdir(self.imgdir):
            self.img = cv2.imread(os.path.join(self.imgdir, each))
            # choose the fit img
            cv2.imshow(self.rotateImgWinname, self.img)
            while True:
                key = cv2.waitKey(10)
                if key == ord('n'):
                    break
                elif key == ord('y'):
                    flagDone = True
                    break
            if flagDone:
                break

        self.imgtoShow = self.img.copy()
        cv2.setMouseCallback(self.rotateImgWinname, self.drawCursortoFindAngle)

        print('click space if you have drawn 2 points to find rotate angle')
        cv2.waitKey(0)
        cv2.destroyWindow(self.rotateImgWinname)

        # need to check the exception if less than 2 pt
        if self.drawCursorCnt >= 2:
            # cal the angles
            dx = self.drawCursorPt[1][0] - self.drawCursorPt[0][0]
            dy = self.drawCursorPt[1][1] - self.drawCursorPt[0][1]

            if dx == 0:
                if dy > 0:
                    self.angle = 90
                elif dy < 0:
                    self.angle = -90
                else:
                    self.angle = 0
            else:
                self.angle = np.degrees(np.arctan(dy / dx))


        # reset the img and imgtoShow
        self.img = None
        self.imgtoShow = None


    def calibratePointer(self):
        # 1. get the pointer calibration img
        flagDone = False    # weather the img is chosen
        print('Calibrate the pointer: press [y/n] to choose one img')
        for each in os.listdir(self.imgdir):
            self.img = cv2.imread(os.path.join(self.imgdir, each))
            cv2.imshow(self.calPointerWinname, self.img)
            while True:
                key = cv2.waitKey(10)             
                if key == ord('n'):
                    break
                elif key == ord('y'):
                    flagDone = True
                    break
            if flagDone:
                break

        # rotate the img
        self.img = rotateImg(self.img, self.angle)
        self.imgtoShow = self.img.copy()
        cv2.imshow(self.calPointerWinname, self.imgtoShow)
        cv2.setMouseCallback(self.calPointerWinname, self.drawPointer)

        print(r'begin to calibrate the pointer')
        print('press [y] if done!')
        print('press [d] if clear!')
        while (1):
            key = cv2.waitKey(10)
            if key == ord('y'):
                break
            elif key == ord('d'):
                # delete all
                self.imgtoShow = self.img.copy()
                self.flagDone = 0
                self.cnt = 0
                cv2.imshow(self.calPointerWinname, self.imgtoShow)


        if self.flagDone == 2:
            result = 'slope = ({:.2f}, {:.2f})'.format(self.slope[0], self.slope[1])
            self.solveBisector()
        else:
            raise Exception('calibrate pointer error')
            
        print('click space when is ok!')
        cv2.waitKey(0)
        cv2.destroyWindow(self.calPointerWinname)

        data = {'slope':self.slope, 'pt':self.pt, 'bisector':self.bisector,
                'angle':self.angle}

        with open(self.save_path, 'wb') as f:
            pk.dump(data, f)

        print('finish calibrate pointer')


    def solveBisector(self):

        if abs(self.slope[0] + self.slope[1]) < 1e-3:
            self.bisector = 0
        else:
            k1 = self.slope[0]
            k2 = self.slope[1]
            temp = 2*(k1*k2-1) / (k1+k2)

            km1 = (temp - (temp ** 2 + 4) ** 0.5) / 2
            km2 = (temp + (temp ** 2 + 4) ** 0.5) / 2

            if abs((k1-km1) / (1 + k1 * km1)) <= 1:
                self.bisector = km1
            else:
                self.bisector = km2

        # visualize the pointer
        x = self.pt[1][0]
        y = self.pt[1][1]
        h, w, _ = self.img.shape

        if self.bisector > 0:
            mind_left = min(x, y, self.len / 2)
            mind_right = min(w-1 - x, h-1 - y, self.len / 2)

            dy0 = mind_left / 2 / ((1 + self.bisector ** 2) ** 0.5)
            dx0 = self.bisector * dy0
            dx0 = round(dx0)
            dy0 = round(dy0)

            dy1 = mind_right / 2 / ((1 + self.bisector ** 2) ** 0.5)
            dx1 = self.bisector * dy1
            dx1 = round(dx1)
            dy1 = round(dy1)

            x0 = x - dx0
            y0 = y - dy0
            x1 = x + dx1
            y1 = y + dy1

        else:
            mind_left = min(x, h-1 - y, self.len / 2)
            mind_right = min(w-1 - x, y, self.len / 2)

            dy0 = mind_left / 2 / ((1 + self.bisector ** 2) ** 0.5)
            dx0 = self.bisector * dy0
            dx0 = round(dx0)
            dy0 = round(dy0)

            dy1 = mind_right / 2 / ((1 + self.bisector ** 2) ** 0.5)
            dx1 = self.bisector * dy1
            dx1 = round(dx1)
            dy1 = round(dy1)

            x0 = x + dx0
            y0 = y + dy0
            x1 = x - dx1
            y1 = y - dy1

        cv2.line(self.imgtoShow, (x0, y0), (x1, y1), gv.blue, 2)
        cv2.imshow(self.calPointerWinname, self.imgtoShow)
        
    
def rotateImg(img, angle):
    r, c, _ = img.shape
    # rotate matrix
    rotateMat = cv2.getRotationMatrix2D((c//2, r//2), angle, 1.0)
    return cv2.warpAffine(img, rotateMat, (c, r))
        
        