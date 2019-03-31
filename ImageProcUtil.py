import cv2
from scipy import ndimage
import numpy as np
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

def cvtcolor_bgr_rgb(img):
    b,g,r=cv2.split(img)
    return cv2.merge((r,g,b))
def threshold_dots(imgarray):
    img = imgarray
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inrangemask=cv2.inRange(hsv, np.array([0,0,160]),np.array([179,255,255]))
    #outrangemask=cv2.inRange(hsv, np.array([11,255,0]),np.array([80,255,160]))
    #inrangemask=cv2.bitwise_not(outrangemask)

    dev,inrangemask=pcv.fill(inrangemask, inrangemask, 400,0)
    inrangemask=ndimage.filters.minimum_filter(inrangemask, (5,5))
    #img = ndimage.maximum_filter(img, size=100)
    hsv=cv2.bitwise_and(hsv, hsv, mask=inrangemask)
    img=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    #img = cv2.medianBlur(img,5)
    middle_sectionx=int(img.shape[0]/2)
    middle_sectiony=int(img.shape[1]/2)
    img=img[(middle_sectionx-200):(middle_sectionx+200), (middle_sectiony-500):(middle_sectiony+500)]
    #Dilation 5x5 kernel 
    kernel = np.ones((5,5),np.uint8)
    img=cv2.dilate(img, kernel, iterations=1)
    #img=cvtcolor_bgr_rgb(img)
    #plt.imshow(img)
    #plt.show()
    
    return img


def threshold_dots_withcenter(imgarray):
    img = imgarray
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inrangemask=cv2.inRange(hsv, np.array([0,0,160]),np.array([179,255,255]))
    #outrangemask=cv2.inRange(hsv, np.array([11,255,0]),np.array([80,255,160]))
    #inrangemask=cv2.bitwise_not(outrangemask)

    dev,inrangemask=pcv.fill(inrangemask, inrangemask, 400,0)
    inrangemask=ndimage.filters.minimum_filter(inrangemask, (5,5))
    #img = ndimage.maximum_filter(img, size=100)
    hsv=cv2.bitwise_and(hsv, hsv, mask=inrangemask)
    img=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    #img = cv2.medianBlur(img,5)
    middle_sectionx=int(img.shape[0]/2)
    middle_sectiony=int(img.shape[1]/2)
    img=img[(middle_sectionx-200):(middle_sectionx+200), (middle_sectiony-500):(middle_sectiony+500)]
    #Dilation 5x5 kernel 
    kernel = np.ones((5,5),np.uint8)
    img=cv2.dilate(img, kernel, iterations=1)
    #img=cvtcolor_bgr_rgb(img)
    #plt.imshow(img)
    #plt.show()
    return tuple([img, tuple([middle_sectionx-200, middle_sectiony-500])])

    
if __name__=='__main__':
    plt.imshow(threshold_dots(cv2.imread("/Users/gghosal/Desktop/gaurav/Plan/PlantCVCroppedTP1/101_2.jpg"))
               )
    plt.show()
