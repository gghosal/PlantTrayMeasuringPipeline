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
    def scan_for_horizontal(self, imagearray):
        """
    
