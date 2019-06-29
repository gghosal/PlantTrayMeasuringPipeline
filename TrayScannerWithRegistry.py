###GeneralMapCreator.py
import DataStructures
import pickle
import csv
import os.path
import cv2
import os
import ImageProcUtil
import HalfPotSegmenter
from plantcv import plantcv as pcv
import numpy as np
from matplotlib import pyplot as plt
import ImageProcUtil
import HalfShelfSegmenter
import NoiseRemoval
from matplotlib import pyplot as plt
import HalfPotSegmenter
import DotCodeReader
import NoiseRemoval
import cv2
import DataStructures
#global TRAY_SECTION
import pickle
def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f
class MapCreator:
    def __init__(self):
        self.shelf_segmenter=HalfShelfSegmenter.HalfShelfSegmenter('/Users/gghosal/Desktop/ProgramFilesPlantPipeline/Vertical.jpg','/Users/gghosal/Desktop/Template.jpg',1400,500)
        self.potsegmenter=HalfPotSegmenter.HalfPotSegmenter()
        self.dotcodereader=DotCodeReader.DotCodeReader("/Users/gghosal/Desktop/dotcodestranslated.dat",{'red':0, "lightblue":98, "darkblue":120, "pink":173, "purple":160})
        self.noise_removal=NoiseRemoval.NoiseRemoval()
        self.registry=dict()
    def save_registry(self, filename):
        file=open(filename, "wb")
        pickle.dump(self.registry,file)
        file.close()
    def load_registry(self, filename):
        file=open(filename, "rb")
        self.registry=pickle.load(file)
        file.close()
    def add_to_registry(self, tray_img_path,section_types):
        os.chdir('/Users/gghosal/Desktop/gaurav/photos')
        tray_counter=0
        tray_struct=list()
        trays=self.shelf_segmenter.split(cv2.imread(tray_img_path))
        for i in trays:
            cleaned=ImageProcUtil.threshold_dots(i)
            cleaned=self.noise_removal.remove_noise(cleaned)
            pots=self.potsegmenter.split_half_trays(i)
            potsfinal=list()
            for q in pots:
                if bool((q.shape[0]>=100) and (q.shape[1]>=100)):
                    potsfinal.append(q)
            pots=potsfinal
            try:
                tray_id=self.dotcodereader.read_image2(cleaned)
                cv2.imshow("hi",i)
                k=cv2.waitKey()
                a=input("Correct Sequence?")
                if not a=="c":
                    dot_description=a.split(",")
                    description=str()
                    for dotqualifier in dot_description:
                        description+=dotqualifier
                    updated_dot_code_id=self.dotcodereader.translator.get(description, "Error!Try again")
                    print("Corrected: ", updated_dot_code_id)
                    tray_id=updated_dot_code_id
                tray_struct.append(tuple(tray_id,section_types[tray_counter%3]))
                tray_counter+=1
            except:pass
            
            self.registry[os.path.split(tray_img_path)[1].split("_")[1]+"_"+os.path.split(tray_img_path)[1].split("_")[2]+os.path.split(tray_img_path)[1].split("_")[3]]=tray_struct
    def read_and_preprocess_image(self, tray_img_path,section_types):
        os.chdir('/Users/gghosal/Desktop/gaurav/photos')
        tray_type_identifier=str(os.path.split(tray_img_path)[1].split("_")[1]+"_"+os.path.split(tray_img_path)[1].split("_")[2]+os.path.split(tray_img_path)[1].split("_")[3])
        tray_struct=list()
        if self.registry.get(tray_type_identifier, "NotHere")=="NotHere":
            self.add_to_registry(tray_img_path, section_types)

        trays=self.shelf_segmenter.split(cv2.imread(tray_img_path))
        traycounter=0
        for i in trays:
            tray_id, sectiontype=self.registry[tray_type_identifier][tray_counter]
            pots=self.potsegmenter.split_half_trays(i)
            for q in pots:
                if bool((q.shape[0]>=100) and (q.shape[1]>=100)):
                    potsfinal.append(q)
            pots=potsfinal
            current_tray_object=DataStructures.Tray(tray_id,section_type)
            current_tray_object.scan_in_pots(pots)
            tray_struct.append(current_tray_object)
            traycounter+=1
        return tray_struct
    
        
            
if __name__=='__main__':
    processing_object=MapCreator()
    os.chdir('/Users/gghosal/Desktop/gaurav/photos')
    for q in listdir_nohidden('/Users/gghosal/Desktop/gaurav/photos'):
        print(q)
        q_filename=int(os.path.split(q)[1].split("_")[3])
        if q_filename==1:
            typeoftray=[2,1,2]
        elif q_filename==2:
            typeoftray=[1,2,1]
        output=processing_object.read_and_preprocess_image(q, typeoftray)
        
    
            
        
            
                                
