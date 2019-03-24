import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
import cv2
import os
import os
import glob


class DotCodeReader:
    """Class which includes the functionality of dotcode reader"""
    def __init__(self, referencedotcodes):
        """Reference dotcodes should be a dictionary including the description strings of the dotcodes, matched with the number of the tray"""
        self.translator=referencedotcodes
    def translate(self, dotdescriptionlist):
        """Dot description list is a sequence of color followed by TRUE/FALSE indicating the presence of a black dot:
            EXAMPLE:
            ["redTRUE", "darkblueFALSE", "pinkTRUE"]"""
        final_str=str()
        for i in dotdescriptionlist:
            finalstr+=i[0]+str(i[1])
        return finalstr
    def 
