import cv2
import numpy as np
import os
import global_vars as gv
import pickle as pk
import time
import matplotlib.pyplot as plt


class CalDigits:

    def __init__(self, imgdir, angle, params_path = None):

        if params_path is None or (not os.path.exists(params_path)):
            self.imgdir = imgdir
            self.calWinName = r'Calibrate-Digits'
            self.angle = angle

            self.cntImg = 0
            self.startCursor = None
            self.endCursor = None
            self.center = []
            self.border = []
            self.poly = None

            self.img = None
            self.imgtoShow = None
            self.lastImgtoShow = None   # undo a time

            self.flagDrawDigit = False
            self.flagDrawBorder = False
            self.save_path = params_path

        else:
            with open(params_path, 'rb') as f:
                self.poly = pk.load(f)



    def drawDigit(self, event, x, y, flags, params):
        if self.flagDrawDigit:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.startCursor = (x, y)
                
            elif (event == cv2.EVENT_MOUSEMOVE) and (flags == cv2.EVENT_FLAG_LBUTTON):   
                self.imgtoShow = self.lastImgtoShow.copy()
                cv2.rectangle(self.imgtoShow, self.startCursor, (x, y), gv.green, 2)
                cv2.imshow(self.calWinName, self.imgtoShow)
            
            elif event == cv2.EVENT_LBUTTONUP:         
                self.endCursor = (x, y)
                tmp = (round((self.startCursor[0]+self.endCursor[0])/2), 
                                    round((self.startCursor[1]+self.endCursor[1])/2))
                self.center.append(tmp)
                
                cv2.circle(self.imgtoShow, self.center[-1], 5, gv.red, 2)
                cv2.imshow(self.calWinName, self.imgtoShow)
                self.lastImgtoShow = self.imgtoShow

                
        if self.flagDrawBorder:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.border.append((x, y))
                cv2.circle(self.lastImgtoShow, (x, y), 5, gv.blue, 2)
                cv2.imshow(self.calWinName, self.lastImgtoShow)


    def processData(self, visualize=True):
        x = []
        y = []

        for i in range(len(self.center)):
            x.append(self.center[i][1])
            y.append(self.border[i][1] - self.center[i][1])

        x = np.array(x)
        y = np.array(y)

        self.poly = np.polyfit(x, y, 2)

        if visualize:
            plt.figure(1)

            v_x = np.linspace(np.min(x), np.max(x), 50)
            v_y = np.polyval(self.poly, v_x)

            plt.scatter(x, y)
            plt.plot(v_x, v_y, 'g')
            plt.title('poly')
        
    def calibrateDigits(self):
        # 1. get a fit photo
        imgList = os.listdir(self.imgdir)

        for each in imgList:
            self.img = cv2.imread(os.path.join(self.imgdir, each))
            self.img = rotateImg(self.img, self.angle)
            cv2.imshow(self.calWinName, self.img)
            
            print('weather the img is fit [y/n]')          
            key = cv2.waitKey(0)
            if key == ord('n'):
                continue

            cv2.setMouseCallback(self.calWinName, self.drawDigit)
            self.imgtoShow = self.img.copy()
            self.lastImgtoShow = self.img.copy()

            time.sleep(0.2)
            # start draw digit
            print('please draw digit!')
            self.flagDrawDigit = True
            self.flagDrawBorder = False
            print('click space when you done')
            cv2.waitKey(0)
            time.sleep(0.2)
            print('please click border!')
            self.flagDrawDigit = False
            self.flagDrawBorder = True
            print('click space when you done')
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # put at the end
            key = input('choose an img or exit calDigit [y/n]: ')
            if key == 'n':
                break
            
        cv2.destroyAllWindows()
        self.processData()

        with open(self.save_path, 'wb') as f:
            pk.dump(self.poly, f)


def rotateImg(img, angle):
    r, c, _ = img.shape
    # rotate matrix
    rotateMat = cv2.getRotationMatrix2D((c // 2, r // 2), angle, 1.0)
    return cv2.warpAffine(img, rotateMat, (c, r))


        
        



