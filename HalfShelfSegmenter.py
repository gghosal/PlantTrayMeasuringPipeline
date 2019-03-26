#Half-Shelf Segmenter
import ImageProcUtil
import cv2
import sys
from detect_peaks import detect_peaks
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

class HalfTraySegmenter:
    """Segments the shelves into individual trays"""
    def __init__(self, trayverticaltemplatefile, trayhorizontaltemplatefile, trayverticaldistance, trayhorizontaldistance):
        """ trayverticaltemplatefile denotes the file containing vertical seperators,
        trayhorizontaltemplatefile denotes file containing horizontal seperators,
        trayverticaldistance denotes approximate distance between pots,
        trayhorizontaldistance denotes approximate distance in length of pots"""
        self.vertical_template=cv2.imread(trayverticaltemplatefile,0)
        self.horizontal_template=cv2.imread(trayhorizontalfile,0)
        self.vertical_distance=trayverticaldistance
        self.horizontal_distance=trayhorizontaldistance
    def scan_along_horizontal(self, imagearray):
        """Find the locations in order to segment the image into pots along the horizontal axis"""
        image_grey=cv2.cvtColor(imagearray, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray,self.horizontal_template,cv2.TM_CCOEFF_NORMED)
        avg_array=np.amax(res,axis=0)
        peaks=detect_peaks(avg_array,mph=0.5,mpd=self.horizontal_distance)
        peaks=np.array(peaks)+int(w/2)
        peaks=list(peaks)
        peaksfinal=list()
        for i in peaks: 
            if (i<=1000) or (i>=9000):
                pass
        else:
            peaksfinal.append(i)
        return peaksfinal
    def split_on_horizontal(self, imagearray):
        """Split along the horizontal axis"""
        peaks=self.scan_along_horizontal(imagearray)
        subsets=list()
        for i in np.split(imagearray, peaks,1):
            subsets.apppend(i)
        return subsets
    def scan_along_vertical(self, imagearray):
        image_grey=cv2.cvtColor(imagearray, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray,self.vertical_template,cv2.TM_CCOEFF_NORMED)
        avg_array=np.amax(res,axis=1)
        peaks=detect_peaks(avg_array,mph=0.5,mpd=self.vertical_distance)
        peaks=np.array(peaks)+int(h/2)
        peaks=list(peaks)
        peaksfinal=list()
        return peaks
        
            
        
        
