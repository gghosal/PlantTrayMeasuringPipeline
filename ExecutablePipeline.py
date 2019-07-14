#####Executable Pipeline


import ImageProcUtil
import HalfShelfSegmenter
import NoiseRemoval
import numpy as np
from matplotlib import pyplot as plt
import HalfPotSegmenter
import DotCodeReader
import time 
import NoiseRemoval
import cv2
import DataStructures
import sys
import warnings
import ErrorHandler
global TRAY_SECTION
import pickle
from plantcv import plantcv as pcv
import os
from skimage.filters import threshold_otsu, threshold_adaptive,threshold_local
global DEBUG
DEBUG=False
class NullDev:
    def write(self,s):
        pass
sys.stderr=NullDev()
##### TRAY SPECIFICATION TYPE(2)->[1,2,1]
##### TRAY SPECIFICATION TYPE(1)->[2,1,2]
#####
### Correction format color1dot1,color2dot2,color3dot3
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

class ExecutablePipeline:
    def __init__(self):
        self.shelf_segmenter=HalfShelfSegmenter.HalfShelfSegmenter('/Users/gghosal/Desktop/UCB/VerticalTemp.jpg','/Users/gghosal/Desktop/Template.jpg',1400,800)
        self.potsegmenter=HalfPotSegmenter.HalfPotSegmenter()
        self.dotcodereader=DotCodeReader.DotCodeReader("/Users/gghosal/Desktop/dotcodestranslated.dat",{'red':0, "lightblue":90, "darkblue":120, "pink":173, "purple":160})
        self.noise_removal=NoiseRemoval.NoiseRemoval()
    def process_tray(self, tray_img_path,section_types):
        """Return list of tray objects"""
        #cropped=ImageProcUtil.crop_out_black(tray_img_path)
        #plt.imshow(cropped)
        #plt.show()
        tray_counter=0
        tray_struct=list()
        trays=self.shelf_segmenter.split(cv2.imread(tray_img_path))
        errors=0
        for i in trays:
            #plt.imshow(i)
            #plt.show()
            try:
                #plt.imshow(i)
                #splt.show()
                pass
            except:
                continue
            #i=self.color_enhancement(i,0.4)
            #plt.imshow(i)
            #plt.show()]
            try:
                #plt.imshow(i)
                #plt.show()
                cleaned,centers_adjustment=ImageProcUtil.threshold_dots3_slack(i)
                cleaned=self.noise_removal.remove_noise(cleaned)
            except Exception as e:
                #print("clean error")
                #print(e)
                continue

            #cleaned=apply_brightness_contrast(cleaned, brightness=0, contrast=69)
            #cleaned_grey=cv2.cvtColor(cleaned, cv2.COLOR_RGB2HSV)
            #cleaned_grey=cleaned_grey
            #cleaned_thresh=cv2.inRange(cleaned_grey,np.array([0,0,150]),np.array([255,255,255]))
            #cleaned=cv2.bitwise_and(cleaned, cleaned, mask=cleaned_thresh)
            #kernel = np.ones((7,7),np.uint8)
            #cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE,(31,31))

            

            
            
            #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            #im = cv2.filter2D(cleaned, -1, kernel)

            #cleaned=cv2.bitwise_and(cleaned, cleaned, mask=th3)
            cleaned=ImageProcUtil.cvtcolor_bgr_rgb(cleaned)
            
            #grey=cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
            #_,thresh=cv2.threshold(grey, 1,255, cv2.THRESH_BINARY)
            #_,binarymask=cv2.threshold(grey, 80,255,cv2.THRESH_BINARY)
            #cleaned=cv2.bitwise_and(cleaned, cleaned, mask=binarymask)
            #binarymask=grey>binarymask

            #print(binarymask)
            
##            for i in range(binarymask.shape[0]):
##                for p in range(binarymask.shape[1]):
##                    if binarymask[i][p]:
##                        binarymask[i][p]=255
##                    else:
##                        binarymask[i][p]=0
##                                            
            #print(binarymask)
            

            #edges=cv2.Canny(cleaned, 200,400)
            
           # dev,cleanedEdge=pcv.fill(edges,edges, 15,0)
            #cleanedEdge=cv2.blur(cleanedEdge,(7,7))
            #circles=cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,50,30,6,15)
            #print(circles)
            #im2, contours, hierarchy=cv2.findContours(cleanedEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            #print(len(contours))
            
            #cleaned_grey=cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype('uint8')
            #print(cleaned_grey)
            #thresholded=cv2.adaptiveThreshold(cleaned,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15, 12)
            #cleaned=cv2.bitwise_and(cleaned,cleaned, mask=thresholded[1])
            #kernel = np.ones((3,3),np.uint8)
            #cleaned = cv2.erode(cleaned,kernel,iterations = 1)
            ###For Dot Detection
            #fordotdetectin
            #device, cleanedfordots=pcv.fill(
            device=0
            #edges=cv2.Canny(cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY),100,200)
            #print(np.shape(edges))
            #device,id_objects, obj_hierarchy = pcv.find_objects(edges,edges,device)
            #clusters_i, contours, hierarchies=pcv.cluster_contours(cleaned, id_objects, obj_hierarchy, 1, 3)
            #print(len(clusters_i))

                
            
 
                       
            #imagew1=cv2.drawContours(cleaned, edged_contours, 0, (255,0,0),5)
        
            
            #plt.imshow(edges)
            
            #plt.show()
            
            #plt.imshow(cleaned)
            #plt.show()
            #print(center)
            #cleaned=self.noise_removal.remove_noise(cleaned
            
 
            #print(i.shape)
            #pots=self.potsegmenter.split_half_trays(i)

            #fig=plt.figure(figsize=(8,8))
            #columns = 4
            #rows = 2
            #counter=1
            potsfinal=list()

            #for pot in pots:
                #cv2.imshow("hi",pot)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                   
            #for j in pots:
                #img = np.random.randint(10, size=(h,w))
                #fig.add_subplot(rows, columns, counter)
                #print(j.shape)
                #plt.imshow(ImageProcUtil.cvtcolor_bgr_rgb(j))
                #plt.show()

            
    
                #plt.imshow(ImageProcUtil.cvtcolor_bgr_rgb(j))
                #plt.show()
            try:
                tray_id=self.dotcodereader.read_image2(cleaned)
                print(tray_id)
                if DEBUG:
                    
                    
                    cv2.imshow("ho",cleaned)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                #cv2.imshow("hi",cleaned)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                centers=self.dotcodereader.get_centers()
                centers_adjusted=list()
                for pq in centers:
                    centers_adjusted.append((pq[0]+centers_adjustment[1], pq[1]+centers_adjustment[0]))
                
                if str(tray_id).isalpha():
                    errors+=1
                    print("unrecognized code")
                    #cv2.imshow("hi",i)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    #print('error')
                    record=ErrorHandler.ErroneousImage(i, "unrecognized_dot_code",tray_id,tray_img_path.split(".")[0]+".shelf",centers_adjusted,section_types[tray_counter%3])
                    error_file_obj=open(str("/Users/gghosal/Desktop/gaurav_new_photos/Errors/"+tray_img_path.split(".")[0]+str(errors)), "wb")
                    pickle.dump(record, error_file_obj)
                    error_file_obj.close()
                    
                    
                    
                    
                #cv2.imshow('detected circles',cleaned)
                #cents=self.dotcodereader.get_centers()
                pots=self.potsegmenter.split_half_trays_with_centers(i,centers_adjusted)
                #centsfinal=list()
                #for pqrs in cents:
                    #centsfinal.append((pqrs[0]+center[0],pqrs[1]+center[1]))
                #print(centsfinal)
                
                #pots=self.potsegmenter.split_half_trays(i)
                for q in pots:
                    if bool((q.shape[0]>=100) and (q.shape[1]>=100)):
                        potsfinal.append(q)
                pots=potsfinal
                #for pot in pots:
                
                    #cv2.imshow("hi", pot)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                #plt.imshow('i',i)
                #plt.show()
                #cv2.imshow("hi",cleaned)
                #k=cv2.waitKey()
                #a=input("Correct Sequence?")
    ##                if not a=="c":
    ##                    dot_description=a.split(",")
    ##                    description=str()
    ##                    for dotqualifier in dot_description:
    ##                        description+=dotqualifier
    ##                    updated_dot_code_id=self.dotcodereader.translator.get(description, "Error!Try again")
    ##                    print("Corrected: ", updated_dot_code_id)
    ##                    tray_id=updated_dot_code_id
    ##                        
                ###th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
                    
                    
                #print(tray_counter%3)
                #q=input("Dot Code")
                current_tray_object=DataStructures.Tray(tray_id,section_types[tray_counter%3])
                current_tray_object.scan_in_pots(pots)
                tray_struct.append(current_tray_object)
                #= #plt.imshow(current_tray_object.get_pot_position(3).get_image())
                #plt.show()
                tray_counter+=1
            except IndexError as e:
                try:
                    print("inner except first")
                    print(e)
                    cleaned,centers_adjustment=ImageProcUtil.threshold_dots3_slack(i)
                    cleaned=self.noise_removal.remove_noise(cleaned)
                    cleaned=ImageProcUtil.cvtcolor_bgr_rgb(cleaned)
                    if DEBUG:
                        cv2.imshow("hi",cleaned)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
        
                    
                    tray_id=self.dotcodereader.read_image2(cleaned)
                    #print(tray_id)
                    centers=self.dotcodereader.get_centers()
                    centers_adjusted=list()
                    for pq in centers:
                        centers_adjusted.append((pq[0]+centers_adjustment[1], pq[1]+centers_adjustment[0]))
                    if str(tray_id).isalpha():
                        #cv2.imshow("hi",i)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                        errors+=1
                        record=ErrorHandler.ErroneousImage(i, "unrecognized_dot_code",tray_id,tray_img_path.split(".")[0]+".shelf",centers_adjusted,section_types[tray_counter%3])
                        error_file_obj=open(str("/Users/gghosal/Desktop/gaurav_new_photos/Errors/"+tray_img_path.split(".")[0]+str(errors)), "wb")
                        pickle.dump(record, error_file_obj)
                        error_file_obj.close()
                        
                        
                        
                        
                    #cv2.imshow('detected circles',cleaned)
                    #cents=self.dotcodereader.get_centers()
                    pots=self.potsegmenter.split_half_trays_with_centers(i,centers_adjusted)
                    #centsfinal=list()
                    #for pqrs in cents:
                        #centsfinal.append((pqrs[0]+center[0],pqrs[1]+center[1]))
                    #print(centsfinal)
                    
                    #pots=self.potsegmenter.split_half_trays(i)
                    for q in pots:
                        if bool((q.shape[0]>=100) and (q.shape[1]>=100)):
                            potsfinal.append(q)
                    pots=potsfinal
                    #for pot in pots:
                    
                        #cv2.imshow("hi", pot)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                    #plt.imshow('i',i)
                    #plt.show()
                    #cv2.imshow("hi",cleaned)
                    #k=cv2.waitKey()
                    #a=input("Correct Sequence?")
        ##                if not a=="c":
        ##                    dot_description=a.split(",")
        ##                    description=str()
        ##                    for dotqualifier in dot_description:
        ##                        description+=dotqualifier
        ##                    updated_dot_code_id=self.dotcodereader.translator.get(description, "Error!Try again")
        ##                    print("Corrected: ", updated_dot_code_id)
        ##                    tray_id=updated_dot_code_id
        ##                        
                    ###th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
                        
                        
                    #print(tray_counter%3)
                    #q=input("Dot Code")
                    current_tray_object=DataStructures.Tray(tray_id,section_types[tray_counter%3])
                    current_tray_object.scan_in_pots(pots)
                    tray_struct.append(current_tray_object)
                    #= #plt.imshow(current_tray_object.get_pot_position(3).get_image())
                    #plt.show()
                        
                    
                    #errors+=1
                    tray_counter+=1
                except Exception as e:
                    errors+=1
                    print(e)
                    #cv2.imshow("hi",cleaned)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    #print("error")
                    record=ErrorHandler.ErroneousImage(i, "misc.error",None,tray_img_path.split(".")[0]+".shelf",None,section_types[tray_counter%3])
                    error_file_obj=open(str("/Users/gghosal/Desktop/gaurav_new_photos/Errors/"+tray_img_path.split(".")[0]+str(errors)), "wb")
                    pickle.dump(record, error_file_obj)
                    error_file_obj.close()
                    print('misc except 1')
                    

                    
                
            except:
                #print(e)
                errors+=1
                print("outer except")
                if DEBUG:
                    cv2.imshow("hi",cleaned)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                #print("error")
                #cv2.imshow("hi",cleaned)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                record=ErrorHandler.ErroneousImage(i, "misc.error",None,tray_img_path.split(".")[0]+".shelf",None,section_types[tray_counter%3])
                error_file_obj=open(str("/Users/gghosal/Desktop/gaurav_new_photos/Errors/"+tray_img_path.split(".")[0]+str(errors)), "wb")
                pickle.dump(record, error_file_obj)
                error_file_obj.close()
        print("error",errors)
        return tray_struct
    def color_enhancement(self,image, factor):
        hsv=cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:,:,2]=2*hsv[:,:,2]
        rgb=cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        r=rgb[:,:,0]
        g=rgb[:,:,1]
        b=rgb[:,:,2]
        final_bgr=cv2.merge([b,g,r])
        return final_bgr
#20131101_Shelf4_1300_1_masked_rotated-1.tif
if __name__=='__main__':
    t=time.time()
    a=ExecutablePipeline()
    for i in list(listdir_nohidden("/Users/gghosal/Desktop/gaurav_new_photos/Shelf51")):
        print(i)
        os.chdir("/Users/gghosal/Desktop/gaurav_new_photos/Shelf51")
        #print(i)
        if int(i.split("_")[3])==1:
            trays=a.process_tray(i,[2,1,2])
        else:
            trays=a.process_tray(i, [1,2,1])
        out_file=open(str("/Users/gghosal/Desktop/gaurav_new_photos/ProgramFiles/"+i.split(".")[0]+".shelf"), "wb")
        pickle.dump(trays, out_file)
        out_file.close()
        #print(time.time()-t)
##    os.chdir("/Users/gghosal/Desktop/gaurav_new_photos/Shelf")
##    trays=a.process_tray("20131104_Shelf3_1900_1_masked.tif",[2,1,2])
#20131027_Shelf3_1200_1_masked.tif
