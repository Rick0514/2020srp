import cv2


img_path = r'../pointer/video2'
neccessary_file = {'calpointer': r'./file/cp.pkl', 'calruler': r'./file/cr.pkl',
                   'caldigits': r'./file/cd.pkl'}

scaling_w = 1
scaling_h = 1.3

#---------------opencv-----------------
white = (255, 255, 255)
black = (0,0,0)
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)

font = cv2.FONT_HERSHEY_SIMPLEX


#---------------east-----------------
east = r'./model/frozen_east_text_detection.pb'

#---------------digitsocr-----------------
do_path = r'./model/best_1.000_acc.ckpt'