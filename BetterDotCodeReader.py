###BetterDotReader
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
class CheckForDots:
    def __init__(self, template_with_black_dot1, template_no_black_dot1, template_with_black_dot2, template_no_black_dot2):
        self.templates_for_dot1=dict({True:template_with_black_dot1, False:template_no_black_dot1})
        self.templates_for_dot2=dict({True:template_with_black_dot2, False:template_no_black_dot1})
    def prepare_templates(self, img_array):
        
    def process_contour_dots_list(self, contourList):
        retlist=list()
        big_dot=contourList[0]
        retlist.append(min([True, False], key=lambda x: cv2.matchShapes(big_dot, self.templates_for_dot1[x])))
        for i in [1,2]:
            dot=contourlist[i]
            retlist.append(min([True, False], key=lambda x: cv2.matchShapes(dot, self.templates_for_dot2[x])))
        return reslist
    
        
            
            
        
    
        
        
