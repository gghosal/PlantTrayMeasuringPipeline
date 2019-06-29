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
class ErroneousImage:
    def __init__(self, image, errorType, guessed_dot_pattern,associated_filename,centers,orientation):
        """image is the image to be used
            errorType={unrecognized_dot_code, dots_missing}
            guessed_dot_pattern:the dot pattern which was read
            20131026_Shelf3_0800_2_masked
            centers-the centers of the dot"""
        self.image=image
        self.centers=centers
        self.pot_segmenter=HalfPotSegmenter.HalfPotSegmenter()
        self.errorType=errorType
    
        self.orientation=orientation
        self.associated_filename=associated_filename
        file_root=os.path.split(associated_filename)[-1].split(".")[0]
        self.date=file_root.split("_")[0]
        self.shelf=str(file_root.split("_")[1]+file_root.split("_")[3])
        self.time=file_root.split("_")[2]
        self.guessed_dot_pattern=guessed_dot_pattern
    def get_image(self):
        return self.image
    def get_errorType(self):
        return self.errorType
    def correct_record(self, name):
        file_obj=open(associated_filename, "rb")
        file_obj.close()
        pots=self.pot_segmenter.split_half_trays_with_centers(self.image, self.centers)
        record_list=pickle.load(file_obj)
        new_record=DataStructures.Tray(name, position)
        new_record.scan_in_pots(pots)
        record_list.append(new_record)
        out_file_obj=open(associated_filename, "wb")
        pickle.dump(record_list, out_file_obj)
        out_file_obj.close()
    
