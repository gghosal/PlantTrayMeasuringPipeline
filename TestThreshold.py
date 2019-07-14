import ImageProcUtil
import HalfShelfSegmenter
import NoiseRemoval
import numpy as np
from matplotlib import pyplot as plt
import HalfPotSegmenter
import DotCodeReader
import NoiseRemoval
import cv2
import math
import DataStructures
global TRAY_SECTION
import pickle
from plantcv import plantcv as pcv
import os
from skimage.filters import threshold_otsu, threshold_adaptive,threshold_local
def apply_brightness_contrast(input_img, brightness = -12, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
def aspect_ratio(cnt):
    #bbox=cv2.boundingRect(cnt)
    pass
    
noise_removal=NoiseRemoval.NoiseRemoval()
shelf_segmenter=HalfShelfSegmenter.HalfShelfSegmenter('/Users/gghosal/Desktop/ProgramFilesPlantPipeline/Vertical2.jpg','/Users/gghosal/Desktop/Template.jpg',1400,900)
for i in list(listdir_nohidden("/Users/gghosal/Desktop/gaurav_new_photos/Shelf62")):
    os.chdir("/Users/gghosal/Desktop/gaurav_new_photos/Shelf62")
    
    for j in shelf_segmenter.split(apply_brightness_contrast(cv2.imread("20131117_Shelf6_0600_2_masked.tif"),brightness=0, contrast=0)):
        try:
            #cleaned,_=ImageProcUtil.threshold_dots3(j)
            #cleaned=noise_removal.remove_noise(cleaned)
            cv2.imshow("hi", j)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cleaned2,_=ImageProcUtil.threshold_dots3_slack(j)
            
            cleaned2=noise_removal.remove_noise(cleaned2)
##            cleaned2grey=cv2.cvtColor(cleaned2, cv2.COLOR_BGR2GRAY)
##            _,cleaned2thresh=cv2.threshold(cleaned2grey, 10, 255, cv2.THRESH_BINARY)
##            #print(cleaned2thresh)
##            w1,cnt,w2=cv2.findContours(cleaned2thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##            #cnt=cnt[1:]
##            #cnt=cnt[0]
##            realcnt=[]
##            for c in cnt:
##                #print(c)
##                perimeter = cv2.arcLength(c, True)
##                area = cv2.contourArea(c)
##                circularity = 4*math.pi*(area/(perimeter*perimeter))
##                print(circularity)
##                if bool((circularity>=0.3) ):
##                    realcnt.append(c)
##                    #print(ar)
##            cnt=np.array(realcnt)
##                
##            #cnt=list(filter(aspect_ratio, np.array(cnt)))
##            cnt=list(sorted(cnt, key=cv2.contourArea, reverse=True))
##            cnt=cnt[0:3]
##            cv2.drawContours(cleaned2,cnt , -1, (0,255,0), 10)
            cv2.imshow("hi", cleaned2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:continue



