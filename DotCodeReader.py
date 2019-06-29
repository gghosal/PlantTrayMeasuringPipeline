import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
import cv2
import os
import glob
import scipy.stats
import statistics
import ImageProcUtil
import pickle
import NoiseRemoval
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

class DotCodeReader:
    """Class which includes the functionality of dotcode reader"""
    def __init__(self, referencedotcodesfile, colordefinitions):
        """Reference dotcodes should be a dictionary including the description strings of the dotcodes, matched with the number of the tray"""
        self.referencedotcodesfile=referencedotcodesfile
        self.translator=pickle.load(open(referencedotcodesfile, "rb"))
        self.colordefinitions=colordefinitions
        self.centers=list()
    def listdir_nohidden(self,path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f
    def translate(self, dotdescriptionlist):
        """Dot description list is a sequence of color followed by TRUE/FALSE indicating the presence of a black dot:
            EXAMPLE:
            ["redTRUE", "darkblueFALSE", "pinkTRUE"]"""
        finalstr=str()
        for i in dotdescriptionlist:
            finalstr+=i[0]+str(i[1])
        return finalstr
    def read_image(self, imagearray):
        """Translate the picture of the dots into a tray number assignment."""
        ###Preprocessing & Object Finding Steps
    
        device, img1gray=pcv.rgb2gray(imagearray,0)
        device, img_binary=pcv.binary_threshold(img1gray, 1, 255, "light", 0)
        device, id_objects, obj_hierarchy=pcv.find_objects(imagearray, img_binary,0)
        device, roi, roi_hierarchy = pcv.define_roi(imagearray, shape = "rectangle", device = device, roi_input = "default",  adjust = False,
                                           x_adj = 600, y_adj = 600, w_adj = 1200, h_adj = 1200)
        device, roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(imagearray, "partial", roi, roi_hierarchy, id_objects, obj_hierarchy, device)
        device, clusters_i, contours, obj_hierarchy = pcv.cluster_contours(device = device, 
                                                                   img = imagearray, 
                                                                   roi_objects = roi_objects, 
                                                                   roi_obj_hierarchy = roi_obj_hierarchy, 
                                                                   nrow = 1, 
                                                                   ncol = int(3))
        out = "/Users/gghosal/Desktop/gaurav/res"
        device, output_path = pcv.cluster_contour_splitimg(device = device, 
                                                       img = imagearray, 
                                                       grouped_contour_indexes = clusters_i,
                                                       contours = contours,
                                                       hierarchy = obj_hierarchy,
                                                       outdir = out)
        obj_hierarchy=obj_hierarchy[0]
        centroids=list()
        totalcontours=list()
        #print(len(contours))
        for clusters1 in clusters_i:
            totalcontour=contours[clusters1[0]]
            for j in clusters1[1:]:
                totalcontour=np.concatenate((contours[j],totalcontour), axis=0)
            totalcontours.append(totalcontour)
            mmt=cv2.moments(totalcontour)
            ycoords=int(mmt['m10']/mmt['m00'])
            xcoords=int(mmt['m01']/mmt['m00']) 
            #cv2.circle(img1, (ycoords, xcoords), 3, (255, 0, 0), 3) 

            centroids.append(tuple([ycoords,xcoords]))
        count11=0
        for clusters1 in clusters_i:
            for contourtocheck in clusters1:
                if not obj_hierarchy[contourtocheck][2] ==-1:
                    if not cv2.contourArea(clusters_i[obj_hierarchy[contourtocheck][2]])>=5:
                        obj_hierarchy[contourtocheck][2]=-1
        dotlist=list()
        for foundcontour in clusters_i:
            hierarchycontours=[obj_hierarchy[j][2] for j in foundcontour]
            if not all([bool(j==-1) for j in hierarchycontours]):
                dotlist.append(True)
            else:
                dotlist.append(False)
        colors=self.read_colors(out)
        for pnr in range(3):
            dot_characteristics.append(tuple((colors[pnr],dotlist[pnr])))
        try:
            return self.translator[self.translate(dot_characteristics)]
        except:
            return "Error"
        
    def read_colors(self, out):
        os.chdir(out)
        colors=list()
        dot_characteristics=list()
        for i in self.listdir_nohidden(out):
            if self.masked(i):
                mask=cv2.imread(i,0)
                unmasked=cv2.imread(i[0:-5])
                width=mask.shape[0]
                length=mask.shape[1]
                color_averagelist=list()
                for w in range(length):
                    for l in range(width):
                        if mask[l][w]==255:
                            color_averagelist.append(img1hsv[l][w][0])
                color_avg=np.median(color_averagelist)
                resultsdict=dict()
                for color in self.colordefinitions:
                    resultsdict[color]=abs(self.colordefinitions[color]-color_avg)
                color=min(resultsdict, key=lambda x:resultsdict[x])
                colors.append(color)
        return colors
    def check_for_dot(self, clusters_i, obj_hierarchy):
        """Check for subcontours indicating dot"""
        #new_obj_hierarchy=obj_hierarchy
        pass
    def masked(self,name):
        return "mask" in name
    def get_centers(self):
        return self.centers
    def read_image2(self, imageread):
        os.chdir('/Users/gghosal/Desktop/gaurav/res/')
        self.centers=list()
        for i in self.listdir_nohidden('/Users/gghosal/Desktop/gaurav/res/'):
            os.remove(i)
        #color_recognition_dict={'red':0, "lightblue":98, "darkblue":120, "pink":175, "purple":140}
        img1=imageread
        device,img1gray=pcv.rgb2gray(img1,0)
        img1hsv=cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        dev,img_binary = pcv.binary_threshold(img1gray, 1, 255, 'light',0)
        device,id_objects, obj_hierarchy = pcv.find_objects(img1, img_binary,0)

        device, roi, roi_hierarchy = pcv.define_roi(img1, shape = "rectangle", device = device, roi_input = "default",  adjust = False,
                                                   x_adj = 600, y_adj = 600, w_adj = 1200, h_adj = 1200)
        device, roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(img1, "partial", roi, roi_hierarchy, id_objects, obj_hierarchy, device)

        device, clusters_i, contours, obj_hierarchy = pcv.cluster_contours(device = device, 
                                                                           img = img1, 
                                                                           roi_objects = roi_objects, 
                                                                           roi_obj_hierarchy = roi_obj_hierarchy, 
                                                                           nrow = 1, 
                                                                           ncol = int(3))

        out = "/Users/gghosal/Desktop/gaurav/res"
        device, output_path = pcv.cluster_contour_splitimg(device = device, 
                                                           img = img1, 
                                                           grouped_contour_indexes = clusters_i,
                                                           contours = contours,
                                                           hierarchy = obj_hierarchy,
                                                           outdir = out)

        obj_hierarchy=obj_hierarchy[0]
        dotQ=list()
        centroids=list()
        totalcontours=list()
        for clusters1 in clusters_i:
            totalcontour=contours[clusters1[0]]
            for j in clusters1[1:]:
                totalcontour=np.concatenate((contours[j],totalcontour), axis=0)
            totalcontours.append(totalcontour)
            mmt=cv2.moments(totalcontour)
            ycoords=int(mmt['m10']/mmt['m00'])
            xcoords=int(mmt['m01']/mmt['m00']) 
            #cv2.circle(img1, (ycoords, xcoords), 3, (255, 0, 0), 3) 
            #plt.imshow(img1)
            #print([ycoords, xcoords])
            #plt.show()
            self.centers.append(tuple([ycoords, xcoords]))
            centroids.append(tuple([xcoords, ycoords]))
        count11=0
        dot_cleaned=apply_brightness_contrast(imageread, brightness=0, contrast=69)
        dot_cleaned_grey=cv2.cvtColor(dot_cleaned, cv2.COLOR_RGB2HSV)
        dot_cleaned_thresh=cv2.inRange(dot_cleaned_grey,np.array([0,0,180]),np.array([255,255,255]))
        dot_cleaned=cv2.bitwise_and(dot_cleaned, dot_cleaned, mask=dot_cleaned_thresh)
        dot_kernel = np.ones((7,7),np.uint8)
        
        #dot_cleaned = cv2.morphologyEx(dot_cleaned, cv2.MORPH_GRADIENT,(5,5))
        #cv2.imshow("hi",dot_cleaned)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        dotQ=self.read_dot3(dot_cleaned)
        #print(dotQ)
##        for clusters1 in clusters_i:
##            for contourtocheck in clusters1:
##                if not obj_hierarchy[contourtocheck][2] ==-1:
##                    if not cv2.contourArea(contours[obj_hierarchy[contourtocheck][2]])>=20: 
##                        obj_hierarchy[contourtocheck][2]=-1
##        counter=0
##        for foundcontour in clusters_i:
##            hierarchycontours=[obj_hierarchy[j][2] for j in foundcontour]
##            #print(hierarchycontours)
##            if not all([bool(j==-1) for j in hierarchycontours]):
##                dotQ.append(True)
##            else:
##                dotQ.append(False)
        os.chdir(out)
        colors=list()
        dot_characteristics=list()
        for i in self.listdir_nohidden(out):
            #print(i)
            if  self.masked(i):
                mask=cv2.imread(i,0)
                unmasked=cv2.imread(i[0:-5])
                width=mask.shape[0]
                length=mask.shape[1]
                color_averagelist=list()
                for w in range(length):
                    for l in range(width):
                        if mask[l][w]==255:
                            color_averagelist.append(img1hsv[l][w][0])
                color_avg=statistics.mode(color_averagelist)
                #print(color_avg)
                resultsdict=dict()
                for color in self.colordefinitions:
                    resultsdict[color]=abs(self.colordefinitions[color]-color_avg)
                color=min(resultsdict, key=lambda x:resultsdict[x])
                colors.append(color)
        #print(colors)
        for pnr in range(3):
            dot_characteristics.append(tuple((colors[pnr],dotQ[pnr])))
        ##print(dot_characteristics)
        try:
            print("Name:", self.translator.get(self.translate(dot_characteristics),self.translate(dot_characteristics)))
            return self.translator.get(self.translate(dot_characteristics),self.translate(dot_characteristics))
        except Exception as e:
            #print(e)l
            pass
    def read_dot2(self, imageread):
        dotQ=list()
        for i in self.centers:
            if np.equal(imageread[i[1]][i[0]],np.array([0,0,0])).all():
                dotQ.append(True)
            else:
                dotQ.append(False)
        return dotQ
    def read_dot3(self, imageread):
        imageread=cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY)
        dotQ=list()
        for i in self.centers:
            dotQ.append(np.equal(imageread[int(i[1]-5):int(i[1]+5), int(i[0]-5):int(i[0]+5)],0).any())
        return dotQ
        
    def read_dot(self, imageread):
        #os.chdir('/Users/gghosal/Desktop/gaurav/res/')
        #self.centers=list()
        #for i in self.listdir_nohidden('/Users/gghosal/Desktop/gaurav/res/'):
            #os.remove(i)
        #color_recognition_dict={'red':0, "lightblue":98, "darkblue":120, "pink":175, "purple":140}
        img1=imageread
        device,img1gray=pcv.rgb2gray(img1,0)
        img1hsv=cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        dev,img_binary = pcv.binary_threshold(img1gray, 1, 255, 'light',0)
        device,id_objects, obj_hierarchy = pcv.find_objects(img1, img_binary,0)

        device, roi, roi_hierarchy = pcv.define_roi(img1, shape = "rectangle", device = device, roi_input = "default",  adjust = False,
                                                   x_adj = 600, y_adj = 600, w_adj = 1200, h_adj = 1200)
        device, roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(img1, "partial", roi, roi_hierarchy, id_objects, obj_hierarchy, device)

        device, clusters_i, contours, obj_hierarchy = pcv.cluster_contours(device = device, 
                                                                           img = img1, 
                                                                           roi_objects = roi_objects, 
                                                                           roi_obj_hierarchy = roi_obj_hierarchy, 
                                                                           nrow = 1, 
                                                                           ncol = int(3))
        dotQ=list()
        obj_hierarchy=obj_hierarchy[0]
        for clusters1 in clusters_i:
            for contourtocheck in clusters1:
                if not obj_hierarchy[contourtocheck][2] ==-1:
                    if not cv2.contourArea(contours[obj_hierarchy[contourtocheck][2]])>=20: 
                        obj_hierarchy[contourtocheck][2]=-1
        counter=0
        for foundcontour in clusters_i:
            hierarchycontours=[obj_hierarchy[j][2] for j in foundcontour]
            #print(hierarchycontours)
            if not all([bool(j==-1) for j in hierarchycontours]):
                dotQ.append(True)
            else:
                dotQ.append(False)
        return dotQ
        

if __name__=='__main__':
    a=DotCodeReader("/Users/gghosal/Desktop/dotcodestranslated.dat",{'red':0, "lightblue":98, "darkblue":120, "pink":173, "purple":160})
    b=NoiseRemoval.NoiseRemoval()
    FILE="/Users/gghosal/Documents/GitHub/PlantTrayMeasuringPipeline/Figure_1.jpg"
    #cv2.imshow("Dotmask",ImageProcUtil.threshold_dots(pcv.readimage(FILE)[0]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(a.read_image2(b.remove_noise(ImageProcUtil.threshold_dots(pcv.readimage(FILE)[0]))))

    
