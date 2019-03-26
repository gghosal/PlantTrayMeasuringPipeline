import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
import cv2
import os
import glob
import ImageProcUtil
import pickle

class HalfPotSegmenter:
    """Uses dots as a reference in order to accurately split the trays"""
    def __init__(self):
        self.initialized=True
