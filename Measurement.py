####MeasurementSystem
import DataStructures
import pickle
import csv
import cv2
import ImageProcUtil
import random
import os.path
from plantcv import plantcv as pcv
import numpy as np
from matplotlib import pyplot as plt
import HalfPotSegmenter
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
class MeasurementSystem:
    """Contains executable code for generating measurments from .shelf files"""
    def __init__(self, measurements_to_create):
        """measurements_to_create refers to a list of measurements to create"""
        self.measurements=measurements_to_create
    def threshold_green(self, image):
        img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        device=0
        green_lower=np.array([30, 120,120]) ##Define lower bound found by experimentation
        green_upper=np.array([90,253,255]) ##Upper bound
        mask=cv2.inRange(img_hsv, green_lower, green_upper)
        #device, mask=pcv.fill(mask, mask, 100, device)
        device, dilated=pcv.dilate(mask, 1,1,device)
        device, dilated=pcv.fill(dilated, dilated, 50, device)
        res=cv2.bitwise_and(img_hsv, img_hsv, mask=dilated)
        #plt.imshow(res)
        #plt.show()
        #cv2.imshow("hi",res)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return dilated,res
    
        
    def process_pot(self,pot_image):
        device=0
        debug=None
        updated_pot_image=self.threshold_green(pot_image)
        #plt.imshow(updated_pot_image)
        #plt.show()
        device, a=pcv.rgb2gray_lab(updated_pot_image, 'a', device)
        device, img_binary=pcv.binary_threshold(a, 127, 255, 'dark', device, None)
        #plt.imshow(img_binary)
        #plt.show()
        mask=np.copy(img_binary)
        device, fill_image=pcv.fill(img_binary, mask, 300, device)
        device, dilated=pcv.dilate(fill_image, 1,1,device)
        device, id_objects, obj_hierarchy=pcv.find_objects(updated_pot_image, updated_pot_image,device)
        device, roi1, roi_hierarchy= pcv.define_roi(updated_pot_image, 'rectangle', device, None, 'default', debug, False)
        device,roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(updated_pot_image, 'partial', roi1, roi_hierarchy, id_objects, obj_hierarchy, device, debug)
        device, obj, mask = pcv.object_composition(updated_pot_image, roi_objects, hierarchy3, device, debug)
        device, shape_header, shape_data, shape_img = pcv.analyze_object(updated_pot_image, "Example1", obj, mask, device, debug, False)
        print(shape_data[1])
    def process_pot2(self, pot_image):
        device=0
        debug=None
        update_pot_image,res=self.threshold_green(pot_image)
        #plt.imshow(update_pot_image)
        #plt.show()
        #device, a=pcv.rgb2gray_lab(update_pot_image, 'a', device)
        #device, img_binary=pcv.binary_threshold(a, 1, 255, 'dark', device, None)
        #plt.imshow(img_binary)
        #plt.show()
        w1, contours, w2 = cv2.findContours(update_pot_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area=0
        
        for i in contours:
            area+=cv2.contourArea(i)
            #cv2.drawContours(img_binary, i, -1, (random.randint(0,255), random.randint(0,255), random.randint(0,255)),1)
        #plt.imshow(img_binary)
        #plt.show()
        return area
            
        
        
        
        
        
        
if __name__=='__main__':
##    r=open("/Users/gghosal/Desktop/gaurav_new_photos/ProgramFiles/20131103_Shelf4_0600_1_masked_rotated.shelf", "rb")
##    pots=pickle.load(r)
##    pot=pots[1]
##    pot=pot.get_all_pots()[0].get_image()
##    plt.imshow(ImageProcUtil.cvtcolor_bgr_rgb(pot))
##    plt.show()
    a=MeasurementSystem("hi")
##    os.chdir("/Users/gghosal/Desktop/gaurav_new_photos/ProgramFiles/")
##    trays_list=pickle.load(open("20131029_Shelf6_0900_1_masked.shelf","rb"))
##    for tray in trays_list:
##           for pot in tray.get_all_pots():
##               cv2.imshow("hi",pot.get_image())
##               cv2.waitKey(0)
##               cv2.destroyAllWindows()
##               pot.store_measurement(a.process_pot2(pot.get_image()))
##               print(pot.output_identifier_csv())
##    print(a.process_pot(pot))
    

    
    for i in listdir_nohidden("/Users/gghosal/Desktop/gaurav_new_photos/ProgramFiles/Shelf61ShelfFiles"):
        #i="20131026_Shelf3_0600_1_masked.shelf"
        os.chdir("/Users/gghosal/Desktop/gaurav_new_photos/ProgramFiles/Shelf61ShelfFiles")
        trays_list=pickle.load(open(i, "rb"))
            #print(trays_list)
            
            
        #segmenter=HalfPotSegmenter.HalfPotSegmenter()
        #pots=segmenter.split_half_trays(r)
        for tray in trays_list:
           for pot in tray.get_all_pots():
               #cv2.imshow("hi",pot.get_image())
               #cv2.waitKey(0)
               #v2.destroyAllWindows()
               pot.store_measurement(a.process_pot2(pot.get_image()))
               print(pot.output_identifier_csv())
        with open(str("/Users/gghosal/Desktop/PlantImagePipeline/Results"+i.split(".")[0]), 'w') as outfile:
            for tray in trays_list:
                for pot in tray.get_all_pots():
                    outfile.write(pot.output_identifier_csv()+"\n")

            

        
        
        
        
