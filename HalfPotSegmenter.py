import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
import random
import cv2
import os
import statistics
import glob
import ImageProcUtil
import pickle
import NoiseRemoval
#
class HalfPotSegmenter:
    """Uses dots as a reference in order to accurately split the trays"""
    def __init__(self):
        self.initialized=True
        self.noise_remover=NoiseRemoval.NoiseRemoval()
    def find_centers_of_dots(self, contours):
        centers=list()
        for i in contours:
            moments=cv2.moments(i)
            ycoords=int(moments['m10']/moments['m00'])
            xcoords=int(moments['m01']/moments['m00'])
            centers.append(tuple([ycoords, xcoords]))
        centers=sorted(centers, key=lambda x: x[0])
        return centers
    def find_dot_locations(self, imagearray):
        imagearraytransformed,cent=ImageProcUtil.threshold_dots_withcenter(imagearray)
        self.cent=cent
        imagearraytransformed=self.noise_remover.remove_noise(imagearraytransformed)
        plt.imshow(imagearraytransformed)
        plt.show()
        img_gray=cv2.cvtColor(imagearraytransformed, cv2.COLOR_BGR2GRAY)
        ret,thresholded=cv2.threshold(img_gray,1,255,cv2.THRESH_BINARY)
        dev,thresholded=pcv.fill(thresholded, thresholded, 5, 0)
        plt.imshow(imagearraytransformed)
        plt.show()
        w1, contours, w2 = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        centers=self.find_centers_of_dots(contours)
        centersfinal=list()
        for i in centers:
            centersfinal.append(tuple([i[0]+self.cent[0], i[1]+self.cent[1]]))
        return centersfinal
    def split_half_trays(self, imagearray):
        centers=self.find_dot_locations(imagearray)
        xsplits=[c[0] for c in centers]
        ysplits=int(statistics.mean([c[1] for c in centers]))
        xsplitted=np.split(imagearray,xsplits,axis=1)
        final=list()
        for j in xsplitted:
            for k in np.split(j, [ysplits], axis=0):
                final.append(k)
        return final
        
        
if __name__=='__main__':
    segmenter=HalfPotSegmenter()
    pots=segmenter.split_half_trays(cv2.imread("/Users/gghosal/Desktop/gaurav/Plan/PlantCVCroppedTP1/126_1.jpg"))
    for j in pots:
        plt.imshow(j)
        plt.show()
    
