import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
import cv2
import os
import glob
import pickle

class DotCodeReader:
    """Class which includes the functionality of dotcode reader"""
    def __init__(self, referencedotcodesfile, colordefinitions):
        """Reference dotcodes should be a dictionary including the description strings of the dotcodes, matched with the number of the tray"""
        self.referencedotcodesfile=referencedotcodesfile
        self.translator=pickle.load(open(referencedotcodesfile, "rb"))
        self.colordefinitions=colordefinitions
    def listdir_nohidden(self,path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f
    def translate(self, dotdescriptionlist):
        """Dot description list is a sequence of color followed by TRUE/FALSE indicating the presence of a black dot:
            EXAMPLE:
            ["redTRUE", "darkblueFALSE", "pinkTRUE"]"""
        final_str=str()
        for i in dotdescriptionlist:
            finalstr+=i[0]+str(i[1])
        return finalstr
    def read_image(self, imagearray):
        """Translate the picture of the dots into a tray number assignment."""
        ###Preprocessing & Object Finding Steps
        device, img1gray=pcv.rgb2gray(imagearray,0)
        device, image_binary=pcv.binary_threshold(img1gray, 1, 255, "light", 0)
        device, id_objects, object_hierarchy=pcv.find_objects(imagearray, img_binary,0)
        device, roi, roi_hierarchy = pcv.define_roi(imgagearray, shape = "rectangle", device = device, roi_input = "default",  adjust = False,
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
                                                       img = img1, 
                                                       grouped_contour_indexes = clusters_i,
                                                       contours = contours,
                                                       hierarchy = obj_hierarchy,
                                                       outdir = out)
        object_hierarchy=object_hierarchy[0]
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

            centroids.append(tuple([xcoords, ycoords]))
        count11=0
        dotlist=self.check_for_dot(clusters_i, obj_hierarchy)
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
        for i in listdir_nohidden(out):
            if  masked(i):
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
    def check_for_dot(self, contourclusters, obj_hierarchy):
        """Check for subcontours indicating dot"""
        #new_obj_hierarchy=obj_hierarchy 
        for clusters1 in clusters_i:
            for contourtocheck in clusters1:
                if not obj_hierarchy[contourtocheck][2] ==-1:
                    if not cv2.contourArea(contours[obj_hierarchy[contourtocheck][2]])>=5:
                        obj_hierarchy[contourtocheck][2]=-1
        dotQ=list()
        for foundcontour in contourcluters:
            hierarchycontours=[obj_hierarchy[j][2] for j in foundcontour]
            if not all([bool(j==-1) for j in hierarchycontours]):
                dotQ.append(True)
            else:
                dotQ.append(False)
        return dotQ


        
