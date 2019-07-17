#TraysPre_Processing
##Example: 20131103_Shelf4_0600_1_masked_rotated.tif
import os
import os.path
import sys

import cv2
import numpy as np

import HalfShelfSegmenter
import ImageProcUtil
import NoiseRemoval

sys.path.append("/Users/gghosal/Desktop/")
from detect_peaks import detect_peaks
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
class FilePreprocesser:
    def __init__(self):
        self.initialized=True
        self.shelf_segmenter=HalfShelfSegmenter.HalfShelfSegmenter('/Users/gghosal/Desktop/ProgramFilesPlantPipeline/Vertical2.jpg','/Users/gghosal/Desktop/Template.jpg',1400,300)
        self.noise_removal=NoiseRemoval.NoiseRemoval()
    def decode_filename(self, filename):
        filename_only=os.path.split(filename)[-1]
        filename_only=filename_only.split(".")[0]
        components=filename_only.split("_")
        output_dict={"Date":components[0], "Shelf":components[1][-1], "Time":components[2], "Row_Number":components[3]}
        return output_dict
    def correct_dimensions(self, image, amount_to_add):
        return cv2.copyMakeBorder(image, top=amount_to_add, bottom=amount_to_add, left=amount_to_add, right=amount_to_add, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    def rotate_image(self, image, degrees):
        M=cv2.getRotationMatrix2D((int(image.shape[0]/2), int(image.shape[1]/2)), degrees,1)
        return cv2.warpAffine(image, M, (int(image.shape[0]), int(image.shape[1])))
    def determine_tray_orientation(self, image):
        grey_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask=cv2.threshold(grey_image, 1,255, cv2.THRESH_BINARY)
        mask=mask[1]
        kernel=np.ones((10,10), np.uint8)
        dilation=cv2.dilate(mask, kernel, iterations=10)
        img,contours,hierarchy = cv2.findContours(dilation, 1, 2)
        max_contour=max(contours, key=cv2.contourArea)
        [vx,vy,x,y]=cv2.fitLine(max_contour, cv2.DIST_L2,0,0.01,0.01)
        return np.sign(vx/vy)
    def rotate_to_horizontal(self, image):
        image=self.correct_dimensions(image, 1600)
        if self.determine_tray_orientation(image)==-1:
            image=self.rotate_image(image, -45)
        elif self.determine_tray_orientation(image)==1:
            image=self.rotate_image(image, 45)
        return ImageProcUtil.crop_out_blackv2(image)
    def find_double_horizontal_line(self, image):
        img_grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_template=cv2.imread("/Users/gghosal/Desktop/UCB/Template.tiff")
        
        image_template=cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_grey,image_template,cv2.TM_CCOEFF_NORMED)
        avg_array=np.amax(res,axis=1)
        peak=np.argmax(avg_array)
        print(peak)
        result=str()
        if (peak<=(img_grey.shape[0]/2)):
            return "TOP"
        else:
            return "BOTTOM"
    def crop_specific(self, image):
        image=image
        img_grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_template=cv2.imread("/Users/gghosal/Desktop/ProgramFilesPlantPipeline/Vertical.jpg")
        
        image_template_2=cv2.imread("/Users/gghosal/Desktop/Template.jpg")
        #plt.imshow(image)
        #plt.show()
        image_template_2=cv2.cvtColor(image_template_2, cv2.COLOR_BGR2GRAY)
        image_template=cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_grey,image_template,cv2.TM_CCOEFF_NORMED)
        avg_array=np.amax(res,axis=1)
        #print(avg_array[(avg_array.shape[0]-350):(avg_array.shape[0])])
        #index2=np.argmax(avg_array[(avg_array.shape[0]-350):(avg_array.shape[0])])+int(avg_array.shape[0]-350)+int(image_template.shape[0])
        index2=detect_peaks(avg_array[(avg_array.shape[0]-350):(avg_array.shape[0])],mph=0.1, mpd=20)[-1]+int(avg_array.shape[0]-350)+int(image_template.shape[0])
        
        #print(index2)
        index1=detect_peaks(avg_array[0:350], mph=0.1,mpd=20)[0]
        index1+=int(image_template.shape[0]/2)
        image=image[index1:index2]
##        res2=cv2.matchTemplate(img_grey,image_template,cv2.TM_CCOEFF_NORMED)
##        avg_array_2=np.amax(res2,axis=0)
##        index_first=np.argmax(avg_array_2[0:150])-image_template_2.shape[1]
##        
##        index_second=np.argmax(avg_array_2[(avg_array_2.shape[0]-150):(avg_array_2.shape[0])])+int(avg_array_2.shape[0]-150)
##        image=image[:, index_first:index_second]
##        print(image.shape)
        return image
###ALONG
        
        
    def determine_dot_code_order(self, image):
        #plt.imshow(image)
        #plt.show()
        trays=self.shelf_segmenter.split(image)
        tray=trays[0]
        #plt.imshow(tray)
        #plt.show()
##        plt.imshow(tray)
##        plt.show()
        tray,_center=ImageProcUtil.threshold_dots3_slack(tray)
        tray=self.noise_removal.remove_noise(tray)
        #tray=self.noise_removal.remove_noise(tray)
        tray_grey=cv2.cvtColor(tray, cv2.COLOR_BGR2GRAY)
        thresholded_tray_grey=cv2.threshold(tray_grey, 1, 255, cv2.THRESH_BINARY)[1]
        #print(thresholded_tray_grey)
        w1, contours, w2=cv2.findContours(thresholded_tray_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        big_dot=max(contours, key=cv2.contourArea)
        moments=cv2.moments(big_dot)
        ycoords=int(moments['m10']/moments['m00'])
        xcoords=int(moments['m01']/moments['m00'])
        #print((xcoords, ycoords))
        #plt.imshow(tray)
        #plt.show()
        if ycoords<=int(tray.shape[1]/2):
            return True
        else:
            return False
    def flip_horizontal_if_necessary(self, image):
        if not self.determine_dot_code_order(image):
            image=cv2.flip(image, 1)
        return image
    def flip_vertical_if_necessary(self, image, row_number):
        if not bool(bool(self.find_double_horizontal_line(image)=="BOTTOM" and row_number==2) or bool(self.find_double_horizontal_line(image)=="TOP" and row_number==1)):
            image=cv2.flip(image,0)
        return image
    def complete_rotation(self, imagename):
        image_metadata=self.decode_filename(imagename)
        image=cv2.imread(imagename)
        image=self.rotate_to_horizontal(image)
        image=self.crop_specific(image)
        image=self.flip_horizontal_if_necessary(image)
        image=self.flip_vertical_if_necessary(image,int(image_metadata["Row_Number"]))
        return image
if __name__=='__main__':
    a=FilePreprocesser()
##    img_out=a.complete_rotation("/Users/gghosal/Desktop/gaurav_new_photos/128-124/20131027_Shelf5_0800_2_masked.tif")
##    plt.imshow(img_out)
##    plt.show()
    for i in listdir_nohidden("/Users/gghosal/Box/r3_orthophotos"):
        #already_processed=[os.path.split(j)[-1].split(".")[0] for j in listdir_nohidden("/Users/gghosal/Desktop/gaurav_new_photos/UndiagonalizedPhotos/")]
        #already_processed2=[os.path.split(j)[-1].split(".")[0] for j in listdir_nohidden("/Users/gghosal/Desktop/gaurav_new_photos/NewUndiagonalizedPhotos/")]
        if (os.path.split(i)[-1].split(".")[0].split("_")[1] == "Shelf6") and (
                os.path.split(i)[-1].split(".")[0].split("_")[3] == "2"):
            try:
                print(i)
            
                outfile=os.path.split(i)
                os.chdir("/Users/gghosal/Box/r3_orthophotos")
                imgout=a.complete_rotation(i)
                #plt.imshow(imgout)
                #plt.show()
                cv2.imwrite(str("/Users/gghosal/Desktop/gaurav_new_photos/Shelf62/" + outfile[1]), imgout)
            except Exception as e:
                print(e)
                pass
    
        else:pass#print("Done")
  

        
        
    #with_line=cv2.line(img,(0,peak), (img.shape[1], peak), (255,0,0),20)
    #print(a.determine_dot_code_order(img))
