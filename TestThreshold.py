import ImageProcUtil
import HalfShelfSegmenter
import NoiseRemoval
import numpy as np
from matplotlib import pyplot as plt
import HalfPotSegmenter
import DotCodeReader
import NoiseRemoval
import cv2
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
noise_removal=NoiseRemoval.NoiseRemoval()
shelf_segmenter=HalfShelfSegmenter.HalfShelfSegmenter('/Users/gghosal/Desktop/ProgramFilesPlantPipeline/Vertical2.jpg','/Users/gghosal/Desktop/Template.jpg',1400,800)
for i in list(listdir_nohidden("/Users/gghosal/Desktop/gaurav_new_photos/UndiagonalizedPhotos")):
    os.chdir("/Users/gghosal/Desktop/gaurav_new_photos/UndiagonalizedPhotos")
    
    for j in shelf_segmenter.split(apply_brightness_contrast(cv2.imread(i),brightness=0, contrast=0)):
        cleaned,_=ImageProcUtil.threshold_dots3(j)
        #cleaned=noise_removal.remove_noise(cleaned)
        cv2.imshow("hi", cleaned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
