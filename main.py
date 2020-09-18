import cv2
import os
import CalPointer as cp
import CalRuler as cr
import CalDigits as cd
import detect_text as dt
import DigitsOcr as do
import numpy as np
import global_vars as gv


def pixel2cm(dpY, inspY, d_poly, up_poly, down_poly):
    # 1. digitpoint y --> baseline y
    bslY = np.polyval(d_poly, dpY) + dpY
    # 2. dif = bslY - inspY --> cm
    dif = bslY - inspY

    if dif > 0:
        cm = 0
        curY = bslY
        tmp_dif = dif
        cnt = 0
        while True and cnt < 100:
            cnt += 1
            upY = np.polyval(up_poly, curY)
            tmp_dif -= upY
            if tmp_dif > 0:
                curY -= upY
                cm -= 1
            else:
                break
        cm -= (tmp_dif / upY + 1)
    else:
        cm = 0
        curY = bslY
        tmp_dif = dif
        cnt = 0
        while True and cnt < 100:
            cnt += 1
            upY = np.polyval(down_poly, curY)
            tmp_dif += upY
            if tmp_dif < 0:
                curY += upY
                cm += 1
            else:
                break
        cm += (tmp_dif / upY - 1)

    return cm


print('Automatic Ruler Reading Programme')
print('Loading the neccessary files...')

# 1. calibrate the pointer
if os.path.exists(gv.neccessary_file['calpointer']):
    calPointer = cp.CalPointer(gv.img_path, params_path=gv.neccessary_file['calpointer'])
else:
    calPointer = cp.CalPointer(gv.img_path, params_path=gv.neccessary_file['calpointer'])
    calPointer.getRotateAngle()
    calPointer.calibratePointer()


# 2. calibrate the ruler
if os.path.exists(gv.neccessary_file['calruler']):
    calRuler = cr.CalRuler(gv.img_path, calPointer.angle, params_path=gv.neccessary_file['calruler'])
else:
    calRuler = cr.CalRuler(gv.img_path, calPointer.angle, params_path=gv.neccessary_file['calruler'])
    cursor = calRuler.calibrateRuler()

# 3. calibrate the digits
if os.path.exists(gv.neccessary_file['caldigits']):
    calDigits = cd.CalDigits(gv.img_path, calPointer.angle, params_path=gv.neccessary_file['caldigits'])
else:
    calDigits = cd.CalDigits(gv.img_path, calPointer.angle, params_path=gv.neccessary_file['caldigits'])
    calDigits.calibrateDigits()

print('all calibration is done \n\n\n')

# prepare the point that pointer and ruler intersect
# (x-x0) / (y-y0) = k1 --- y = k2x + b
insPoint = None
k1 = calPointer.bisector
k2, b2 = calRuler.border_poly
x0, y0 = calPointer.pt[1]
if k1 == 0:
    insPoint = (round(x0), round(k2*x0 + b2))
else:
    x = (b2 - y0 + x0/k1) / (1/k1 - k2)
    insPoint = (round(x), round(k2*x + b2))

# 4. prepare the east model
print('load the detect text model')
textModel = dt.DetectText(gv.east)
digitsOcr = do.DigitsOcr(gv.do_path)


# 5. traverse the img in file or capture the video in real-time
#
# img = cv2.imread(os.path.join(gv.img_path, '124.jpg'))
# img = cp.rotateImg(img, calPointer.angle)
# img = cv2.GaussianBlur(img, (5, 5), 1.0)
#
# cv2.imshow('1', img)
# if cv2.waitKey(0) == ord('y'):
#     boxes = textModel.detect_text(img)
# cv2.destroyAllWindows()

wrongpics = []
problempic = []
for each in os.listdir(gv.img_path):
    # 5.1 read the img
    img = cv2.imread(os.path.join(gv.img_path, each))
    img = cp.rotateImg(img, calPointer.angle)
    img = cv2.medianBlur(img, 5)
    # 5.2 detect the text in img
    try:
        boxes = textModel.detect_text(img)

        # 5.3 select one box
        if boxes.shape[0] == 0:
            print('no boxes is detected')
            continue
        else:
            # pick the box with center is cloest to pt[1]
            flag = True
            for (x, y, ex, ey) in boxes:
                center = (y+ey)/2
                dist = abs(center - calPointer.pt[1][1])
                if flag:
                    flag = False
                    min_dist = dist
                    pick = (x, y, ex, ey)
                else:
                    if dist < min_dist:
                        min_dist = dist
                        pick = (x, y, ex, ey)

            # scaling the pickbox
            w, h = pick[2] - pick[0], pick[3] - pick[1]
            center = ((pick[0] + pick[2])/2, (pick[1] + pick[3])/2)
            w = int(gv.scaling_w * w)
            h = int(gv.scaling_h * h)
            pick = (int(center[0] - w/2), int(center[1] - h/2),
                    int(center[0] + w/2), int(center[1] + h/2))

            # 5.4 read the number from the box picked
            imgPick = img[pick[1]:pick[3], pick[0]:pick[2]]
            # try:
            rects, preds = digitsOcr.ocr_img(imgPick)
            num, _ = digitsOcr.getDigit(rects, preds)

            # 5.5 cal the real number

            cm = pixel2cm((pick[1] + pick[3])/2, insPoint[1],
                          calDigits.poly, calRuler.up_poly, calRuler.down_poly)
            num += 0.1 * cm
            strnum = '{:.2f}'.format(num)
            # except Exception as e:
            #     if e.args == 'PredError':
            #         continue
            # num, imgPick = digitsOcr.getDigit(rects, preds, visualize=True, img=imgPick)
            # if num / 100 < 1:
            #     problempic.append(each)
            # cv2.imshow('1', imgPick)

            cv2.circle(img, insPoint, 3, gv.white, 1)
            cv2.rectangle(img, pick[:2], pick[2:], gv.green, 2)
            cv2.putText(img, strnum, pick[:2], gv.font, 1, gv.blue, 2)
            cv2.imshow('2', img)
            key = cv2.waitKey(500)
            if key == 27:
                cv2.destroyAllWindows()
                break

    except IndexError:
        wrongpics.append(each)
        continue






