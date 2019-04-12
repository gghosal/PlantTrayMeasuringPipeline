#Half-Shelf Segmenter
import ImageProcUtil
import cv2
import sys
sys.path.append("/Users/gghosal/Desktop/")
from detect_peaks import detect_peaks
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

class HalfShelfSegmenter:
    """Segments the shelves into individual trays"""
    def __init__(self, trayverticaltemplatefile, trayhorizontaltemplatefile, trayverticaldistance, trayhorizontaldistance):
        """ trayverticaltemplatefile denotes the file containing vertical seperators,
        trayhorizontaltemplatefile denotes file containing horizontal seperators,
        trayverticaldistance denotes approximate distance between pots,
        trayhorizontaldistance denotes approximate distance in length of pots"""
        self.vertical_template=cv2.imread(trayverticaltemplatefile,0)
        self.horizontal_template=cv2.imread(trayhorizontaltemplatefile,0)
        self.vertical_distance=trayverticaldistance
        self.horizontal_distance=trayhorizontaldistance
    def scan_along_horizontal(self, imagearray):
        """Find the locations in order to segment the image into pots along the horizontal axis"""
        image_grey=cv2.cvtColor(imagearray, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(image_grey,self.horizontal_template,cv2.TM_CCOEFF_NORMED)
        avg_array=np.amax(res,axis=0)
        peaks=detect_peaks(avg_array,mph=0.5,mpd=self.horizontal_distance)
        peaks=np.array(peaks)+int(self.horizontal_template.shape[::-1][0]/2)
        peaks=list(peaks)
        peaksfinal=list()
        for i in peaks: 
            if (i<=1000) or (i>=9000):
                pass
            else:
                peaksfinal.append(i)
        return peaksfinal
    def split_on_horizontal(self, imagearray):
        """Split along the horizontal axis"""
        peaks=self.scan_along_horizontal(imagearray)
        subsets=list()
        for i in np.split(imagearray, peaks,1):
            subsets.append(i)
        return subsets
    def check_pots(self, splits):
        counter=0
        newsplits=list()
        for j in splits:
            if j.shape[0]>=900:
                #del splits[contour]
                newsplits.append(j[0:int(j.shape[0]/2)])
                newsplits.append(j[int(j.shape[0]/2):j.shape[0]])
            else:
                newsplits.append(j)
        return newsplits
    def scan_along_vertical(self, imagearray):
        image_grey=cv2.cvtColor(imagearray, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(image_grey,self.vertical_template,cv2.TM_CCOEFF_NORMED)
        avg_array=np.amax(res,axis=1)
        peaks=detect_peaks(avg_array,mph=0.3,mpd=self.vertical_distance)
        peaks=np.array(peaks)+int(self.vertical_template.shape[::-1][1]/2)
        peaks=list(peaks)
        peaksfinal=list()
        return peaks
    def split_along_vertical(self, imagearray,tosplit):
        peak=self.scan_along_vertical(imagearray)
        subsets=list()
        for i in tosplit:
            for j in np.split(i,peak,0):
                subsets.append(j)
        return subsets
    def split(self, imagearray):
        horizontal_split=self.split_on_horizontal(imagearray)
        final_split=self.split_along_vertical(imagearray,horizontal_split)
        final_split=self.check_pots(final_split)
        return final_split
if __name__=='__main__':
    segmenter=HalfShelfSegmenter('/Users/gghosal/Desktop/ProgramFilesPlantPipeline/Vertical.jpg','/Users/gghosal/Desktop/Template.jpg',1400,500)
    #cv2.imshow("Cropped", ImageProcUtil.crop_out_black('/Users/gghosal/Desktop/gaurav_new_photos/20131105_Shelf4_0600_1_masked_rotated.tif'))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(ImageProcUtil.crop_out_black('/Users/gghosal/Desktop/gaurav_new_photos/20131104_Shelf4_0600_1_masked_rotated.tif').shape)
    pots=segmenter.split(ImageProcUtil.crop_out_black('/Users/gghosal/Desktop/gaurav_new_photos/20131104_Shelf4_0600_1_masked_rotated.tif'))
    for i in pots:
        cv2.imshow("Pot",i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
            
        
        
