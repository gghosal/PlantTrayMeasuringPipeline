import math

import cv2
import numpy as np
from plantcv import plantcv as pcv


class NoiseRemoval:
    """Uses dots as a reference in order to accurately split the trays"""
    def __init__(self):
        self.initialized=True
    def circularity(self,x):
            perimeter = cv2.arcLength(x, True)
            area = cv2.contourArea(x)
            try:
                return 4*math.pi*(area/(perimeter*perimeter))
            except:return 0
        
    def sort_contours_by_area(self, contours):
        return sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    def sort_contours_by_circularity(self, contours):
        return sorted(contours, key=lambda x: self.circularity(x), reverse=True)

    def remove_green(self, imagearray):
        """Expects a dot thresholded"""
        hsv=cv2.cvtColor(imagearray, cv2.COLOR_BGR2HSV)
        green_lower=np.array([20, 0,0])
        green_upper=np.array([90,255,255])
        mask=cv2.inRange(hsv, green_lower, green_upper
                )
        mask=cv2.bitwise_not(mask)
        device,mask=pcv.fill(mask,mask,500,0)
        device,mask=pcv.fill(mask,mask,500,0)
        res=cv2.bitwise_and(imagearray, imagearray, mask=mask
                   )
        return res
    def remove_noise(self, imagearray):
        #plt.imshow(imagearray)
        #plt.show()
        imagearraytransformed=imagearray
        #imagearraytransformed=ImageProcUtil.threshold_dots(imagearray)
        #imagearraytransformed=self.remove_green(imagearraytransformed)
        #plt.imshow(imagearraytransformed)
        #plt.show()
        img_gray=cv2.cvtColor(imagearraytransformed, cv2.COLOR_BGR2GRAY)
        ret, thresholded = cv2.threshold(img_gray, 1, 255, cv2.THRESH_TRUNC)
        device,thresholded=pcv.fill(thresholded,thresholded,400,0)
        w1, contours, w2 = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        realcnt=[]
        for c in contours:
            #print(c)
            perimeter = cv2.arcLength(c, True)
            x,y,w,h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            try:
                circularity = 4*math.pi*(area/(perimeter*perimeter))
            except:continue

            #print(circularity)
            if bool((float(w/h)>=0.5)):
                realcnt.append(c)
                #print(ar)
        contours=np.array(realcnt)
        #cnt=
        #final_contours=list()
        #for i in range(len(w2)):
            #if w2[0][i][-1]==-1:
               # final_contours.append(contours[i])
        #contours=final_contours
        #print(len(contours))
        contours_sorted=self.sort_contours_by_area(contours)
        contours_sorted=contours_sorted[0:3]
        mask = np.zeros(imagearraytransformed.shape[0:2], np.uint8)
        for i in range(len(contours_sorted)):
            cv2.drawContours(mask, contours_sorted, i, (255),-1)#(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        
        final=cv2.bitwise_and(imagearraytransformed, imagearraytransformed, mask=mask)
        #cv2.imshow('detected circles',final)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return final
