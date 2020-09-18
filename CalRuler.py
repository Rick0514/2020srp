import cv2
import numpy as np
import os
import global_vars as gv
import pickle as pk
from operator import itemgetter
import matplotlib.pyplot as plt

class CalRuler:

    def __init__(self, imgdir, angle, params_path=None):

        if params_path is None or (not os.path.exists(params_path)):
            self.imgdir = imgdir
            self.calWinName = r'Calibrate-Ruler'
            self.cursorPath = r'./cursor'
            self.cntImg = 0

            self.angle = angle
            self.img = None
            self.imgtoShow = None
            self.lastImgtoShow = None   # undo a time

            self.flagDone = False
            self.cursor = []
            self.xycursor = []
            self.up = []
            self.down = []
            self.up_poly = None
            self.down_poly = None
            self.border_poly = None

            # set up a stoplist for dir traversed
            self.stoplist = []
            self.save_path = params_path
        else:
            with open(params_path, 'rb') as f:
                data = pk.load(f)
            self.up_poly = data['up_poly']
            self.down_poly = data['down_poly']
            self.border_poly = data['border_poly']

    def drawCursor(self, event, x, y, flags, params):
        if not self.flagDone:
            if event == cv2.EVENT_LBUTTONUP:
                self.lastImgtoShow = self.imgtoShow.copy()
                cv2.circle(self.imgtoShow, (x,y), 5, gv.white, 2)
                cv2.imshow(self.calWinName, self.imgtoShow)
                self.cursor.append((x, y))

    
    def reset(self):
        
        self.img = None
        self.imgtoShow = None
        self.lastImgtoShow = None   

        self.flagDone = False
        self.cursor = []
        
    
    def processCursor(self):

        # sort the cursor first: small--big
        self.cursor = sorted(self.cursor, key=itemgetter(1))

        for i in range(len(self.cursor)-1):
            self.xycursor.append(self.cursor[i])
            tmp = self.cursor[i+1][1] - self.cursor[i][1]
            self.down.append((self.cursor[i][1], tmp))
            self.up.append((self.cursor[i+1][1], tmp))
    
    
    def getCursor(self):
        # 1. get a fit photo
        imgList = os.listdir(self.imgdir)
        flagPhotoDone = False
        print('Calibrate the ruler: press [y/n] to choose one')
        for each in imgList:
            if each in self.stoplist:
                continue
            
            self.img = cv2.imread(os.path.join(self.imgdir, each))

            while(1):
                cv2.imshow(self.calWinName, self.img)
                key = cv2.waitKey(10)
                if key == ord('n'):
                    break
                elif key == ord('y'):
                    flagPhotoDone = True
                    self.stoplist.append(each)
                    break

            if(flagPhotoDone):
                break
        
        self.cntImg += 1
        
        print('Please find the cursor and click it!')
        print('press y --> done\npress z --> undo one')
        print('press u --> undo all')

        self.img = rotateImg(self.img, self.angle)
        cv2.imshow(self.calWinName, self.img)
        # allow once undo and undo all
        cv2.setMouseCallback(self.calWinName, self.drawCursor)
        self.imgtoShow = self.img.copy()
        self.lastImgtoShow = self.imgtoShow.copy()
        while(1):
            key = cv2.waitKey(10)

            if key == ord('y'):
                self.flagDone = True
                break
            elif key == ord('z'):   # undo a time
                del(self.cursor[-1])
                self.imgtoShow = self.lastImgtoShow
                cv2.imshow(self.calWinName, self.imgtoShow)
            elif key == ord('u'):   # undo all
                self.cursor = []
                self.imgtoShow = self.img.copy()
                self.lastImgtoShow = self.imgtoShow.copy()
                cv2.imshow(self.calWinName, self.imgtoShow)
        
        cv2.destroyAllWindows()
        self.processCursor()
        self.reset()
        
    def fitmodel(self, visualize=True):
        times = 2

        n = len(self.up)
        xup, yup = np.zeros(n), np.zeros(n)
        xdown, ydown = np.zeros(n), np.zeros(n)
        nn = len(self.xycursor)
        xb, yb = np.zeros(nn), np.zeros(nn)

        for i in range(n):
            xup[i], yup[i] = self.up[i]
            xdown[i], ydown[i] = self.down[i]

        for i in range(nn):
            xb[i], yb[i] = self.xycursor[i]

        self.up_poly = np.polyfit(xup, yup, times)
        self.down_poly = np.polyfit(xdown, ydown, times)
        self.border_poly = np.polyfit(xb, yb, 1)

        if visualize:
            v_xup = np.linspace(np.min(xup), np.max(xup), 50)
            v_yup = np.polyval(self.up_poly, v_xup)
            v_xdown = np.linspace(np.min(xdown), np.max(xdown), 50)
            v_ydown = np.polyval(self.down_poly, v_xdown)
            v_xb = np.linspace(np.min(xb), np.max(xb), 50)
            v_yb = np.polyval(self.border_poly, v_xb)

            plt.subplot(131)
            plt.scatter(xup, yup)
            plt.plot(v_xup, v_yup, 'g')
            plt.title('up_poly')

            plt.subplot(132)
            plt.scatter(xdown, ydown)
            plt.plot(v_xdown, v_ydown, 'g')
            plt.title('down_poly')

            plt.subplot(133)
            plt.scatter(xb, yb)
            plt.plot(v_xb, v_yb, 'g')
            plt.title('border_poly')


    def calibrateRuler(self):

        # 1. get cursor repeatedly 
        while(1):
            self.getCursor()
            key = input('continue to get cursor [y/n]: ')
            if(key == 'n'):
                break

        # 2. fit the polynomial
        self.fitmodel(visualize=True)

        # 3. save data
        data = {'border_poly': self.border_poly, 'up_poly': self.up_poly,
                'down_poly': self.down_poly}

        with open(self.save_path, 'wb') as f:
            pk.dump(data, f)

        print('finish calibrate ruler')

def rotateImg(img, angle):
    r, c, _ = img.shape
    # rotate matrix
    rotateMat = cv2.getRotationMatrix2D((c // 2, r // 2), angle, 1.0)
    return cv2.warpAffine(img, rotateMat, (c, r))

