import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
import random
import cv2
import os
import glob
import ImageProcUtil
import pickle

class NoiseRemoval:
    """Uses dots as a reference in order to accurately split the trays"""
    def __init__(self):
        self.initialized=True
    def sort_contours_by_area(self, contours):
        return sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    def remove_noise(self, imagearray):
        imagearraytransformed=ImageProcUtil.threshold_dots(imagearray)
        img_gray=cv2.cvtColor(imagearraytransformed, cv2.COLOR_BGR2GRAY)
        ret,thresholded=cv2.threshold(img_gray,1,255,cv2.THRESH_BINARY)
        w1, contours, w2 = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_sorted=self.sort_contours_by_area(contours)
        contours_sorted=contours_sorted[0:3]
        mask = np.zeros(imagearraytransformed.shape[0:2], np.uint8)
        for i in range(len(contours_sorted)):
            cv2.drawContours(mask, contours_sorted, i, (255),-1)#(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        final=cv2.bitwise_and(imagearraytransformed, imagearraytransformed, mask=mask)
        #cv2.imshow('detected circles',final)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return final
