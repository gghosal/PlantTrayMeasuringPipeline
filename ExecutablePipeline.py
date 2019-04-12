#####Executable Pipeline


import ImageProcUtil
import HalfShelfSegmenter
import NoiseRemoval
from matplotlib import pyplot as plt
import HalfPotSegmenter
import DotCodeReader
import NoiseRemoval
import cv2
class ExecutablePipeline:
    def __init__(self):
        self.shelf_segmenter=HalfShelfSegmenter.HalfShelfSegmenter('/Users/gghosal/Desktop/ProgramFilesPlantPipeline/Vertical.jpg','/Users/gghosal/Desktop/Template.jpg',1400,500)
        self.potsegmenter=HalfPotSegmenter.HalfPotSegmenter()
        self.dotcodereader=DotCodeReader.DotCodeReader("/Users/gghosal/Desktop/dotcodestranslated.dat",{'red':0, "lightblue":98, "darkblue":120, "pink":173, "purple":160})
        self.noise_removal=NoiseRemoval.NoiseRemoval()
    def process_tray(self, tray_img_path):
        cropped=ImageProcUtil.crop_out_black(tray_img_path)
        #plt.imshow(cropped)
        #plt.show()
        trays=self.shelf_segmenter.split(cropped)
        for i in trays:
            #cleaned=ImageProcUtil.threshold_dots(i)
            #cleaned=self.noise_removal.remove_noise(cleaned)
            #cv2.imshow('detected circles',cleaned)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #plt.imshow(ImageProcUtil.cvtcolor_bgr_rgb(i))
            #plt.show()
            try:
                #print(self.dotcodereader.read_image2(cleaned))
                pots=self.potsegmenter.split_half_trays(i)
                fig=plt.figure(figsize=(8,8))
                columns = 4
                rows = 2
                counter=1
                for j in pots:
                    #img = np.random.randint(10, size=(h,w))
                    fig.add_subplot(rows, columns, counter)
                    plt.imshow(ImageProcUtil.cvtcolor_bgr_rgb(j))
                    counter+=1
                plt.show()
                plt.cla()
                plt.clf()
            
    
                #plt.imshow(ImageProcUtil.cvtcolor_bgr_rgb(cleaned))
                #plt.show()
            except:
                #cv2.imshow('detected circles',cleaned)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                continue
if __name__=='__main__':
    a=ExecutablePipeline()
    a.process_tray('/Users/gghosal/Desktop/gaurav_new_photos/20131103_Shelf4_0600_1_masked_rotated.tif')
            
