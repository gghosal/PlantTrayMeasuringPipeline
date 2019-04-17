#####Executable Pipeline


import ImageProcUtil
import HalfShelfSegmenter
import NoiseRemoval
from matplotlib import pyplot as plt
import HalfPotSegmenter
import DotCodeReader
import NoiseRemoval
import cv2
import DataStructures
global TRAY_SECTION
import pickle

##### TRAY SPECIFICATION TYPE(2)->[1,2,1]
##### TRAY SPECIFICATION TYPE(1)->[2,1,2]
##### 


class ExecutablePipeline:
    def __init__(self):
        self.shelf_segmenter=HalfShelfSegmenter.HalfShelfSegmenter('/Users/gghosal/Desktop/ProgramFilesPlantPipeline/Vertical.jpg','/Users/gghosal/Desktop/Template.jpg',1400,500)
        self.potsegmenter=HalfPotSegmenter.HalfPotSegmenter()
        self.dotcodereader=DotCodeReader.DotCodeReader("/Users/gghosal/Desktop/dotcodestranslated.dat",{'red':0, "lightblue":98, "darkblue":120, "pink":173, "purple":160})
        self.noise_removal=NoiseRemoval.NoiseRemoval()
    def process_tray(self, tray_img_path,section_types):
        """Return list of tray objects"""
        #cropped=ImageProcUtil.crop_out_black(tray_img_path)
        #plt.imshow(cropped)
        #plt.show()
        tray_counter=0
        tray_struct=list()
        trays=self.shelf_segmenter.split(cv2.imread(tray_img_path))
        for i in trays:
            cleaned=ImageProcUtil.threshold_dots(i)
            cleaned=self.noise_removal.remove_noise(cleaned)
            #cv2.imshow('detected circles',cleaned)
            
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #plt.imshow(ImageProcUtil.cvtcolor_bgr_rgb(i))
            #plt.show()
 
            #print(i.shape)
            pots=self.potsegmenter.split_half_trays(i)
            #plt.imshow(i)
            #plt.show()
            #fig=plt.figure(figsize=(8,8))
            #columns = 4
            #rows = 2
            #counter=1
            potsfinal=list()
            for q in pots:
                if bool((q.shape[0]>=100) and (q.shape[1]>=100)):
                    potsfinal.append(q)
            pots=potsfinal
                   
            for j in pots:
                #img = np.random.randint(10, size=(h,w))
                #fig.add_subplot(rows, columns, counter)
                #print(j.shape)
                #plt.imshow(ImageProcUtil.cvtcolor_bgr_rgb(j))
                #plt.show()

            
    
                plt.imshow(ImageProcUtil.cvtcolor_bgr_rgb(j))
                plt.show()
            try:
                tray_id=self.dotcodereader.read_image2(cleaned)
                print(tray_counter%3)
                current_tray_object=DataStructures.Tray(tray_id,section_types[tray_counter%3])
                current_tray_object.scan_in_pots(pots)
                tray_struct.append(current_tray_object)
                #= #plt.imshow(current_tray_object.get_pot_position(3).get_image())
                #plt.show()
                tray_counter+=1
            except Exception as e :
                #cv2.imshow('detected circles',cleaned)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                print(e)
                tray_counter+=1
                continue
        return tray_struct
if __name__=='__main__':
    a=ExecutablePipeline()
    trays=a.process_tray('/Users/gghosal/Desktop/gaurav/photos/20131108_Shelf4_1300_1_masked_rotated.tif',[1,2,1])
    out_file=open("/Users/gghosal/Desktop/gaurav_new_photos/ProgramFiles/20131108_Shelf4_1300_1_masked_rotated.shelf", "wb")
    pickle.dump(trays, out_file)
    out_file.close()
            
