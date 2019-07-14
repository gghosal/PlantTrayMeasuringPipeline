import cv2
from scipy import ndimage
import numpy as np
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import NoiseRemoval
def cvtcolor_bgr_rgb(img):
    b,g,r=cv2.split(img)
    return cv2.merge((r,g,b))
def threshold_dots4(imgarray):
    img = imgarray
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    outrangemask=cv2.inRange(hsv, np.array([20,0,0]),np.array([70,255,255]))

    #outrangemask=cv2.inRange(hsv, np.array([11,255,0]),np.array([80,255,160]))
    inrangemask=cv2.bitwise_not(outrangemask)
    #hsv1=cv2.bitwise_and(hsv,hsv, mask=inrangemask)
    #inrangemask = cv2.bilateralFilter(inrangemask,9,75,75)
    inrangemask=ndimage.filters.minimum_filter(inrangemask, (2,2))
    
    dev,inrangemask=pcv.fill(inrangemask, inrangemask, 300,0)
    
    #img = ndimage.maximum_filter(img, size=100)
    hsv=cv2.bitwise_and(hsv, hsv, mask=inrangemask)

    inrangemask2=cv2.inRange(hsv,np.array([0,0,135]), np.array([255,255,255]))
    #inrangemask2 = cv2.bilateralFilter(inrangemask2,9,75,75)
    hsv=cv2.bitwise_and(hsv,hsv, mask=inrangemask2)
    
    img=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = cv2.bilateralFilter(img,5,200,200)

    img = cv2.medianBlur(img,5)
    middle_sectionx=int(img.shape[0]/2)
    middle_sectiony=int(img.shape[1]/2)
    img=img[(middle_sectionx-200):(middle_sectionx+200), (middle_sectiony-500):(middle_sectiony+500)]
    #Dilation 5x5 kernel
    
    #kernel = np.ones((3,3),np.uint8)
    #img=cv2.dilate(img, kernel, iterations=1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    #img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    img=cvtcolor_bgr_rgb(img)
    #plt.imshow(img)
    #plt.show()
    
    return img
def threshold_dots3(imgarray):
    img = imgarray
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    outrangemask=cv2.inRange(hsv, np.array([8,0,0]),np.array([90,255,255]))

    #outrangemask=cv2.inRange(hsv, np.array([11,255,0]),np.array([80,255,160]))
    inrangemask=cv2.bitwise_not(outrangemask)
    #hsv1=cv2.bitwise_and(hsv,hsv, mask=inrangemask)
    #inrangemask = cv2.bilateralFilter(inrangemask,9,75,75)
    #inrangemask=ndimage.filters.minimum_filter(inrangemask, (2,2))
    
    #dev,inrangemask=pcv.fill(inrangemask, inrangemask, 700,0)
    
    #img = ndimage.maximum_filter(img, size=100)
    hsv=cv2.bitwise_and(hsv, hsv, mask=inrangemask)

    inrangemask2=cv2.inRange(hsv,np.array([0,0,150]), np.array([255,255,255]))
    #inrangemask2 = cv2.bilateralFilter(inrangemask2,9,75,75)
    hsv=cv2.bitwise_and(hsv,hsv, mask=inrangemask2)
    inrangemask3=cv2.inRange(hsv,np.array([0,40,0]), np.array([255,255,255]))
    hsv=cv2.bitwise_and(hsv,hsv,mask=inrangemask3)
    
    img=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #img = cv2.bilateralFilter(img,5,200,200)

    img = cv2.medianBlur(img,5)
    middle_sectionx=int(img.shape[0]/2)
    middle_sectiony=int(img.shape[1]/2)
    img=img[(middle_sectionx-150):(middle_sectionx+150), (middle_sectiony-500):(middle_sectiony+500)]
    #Dilation 5x5 kernel
    
    #kernel = np.ones((3,3),np.uint8)
    #img=cv2.dilate(img, kernel)
    #img=cv2.dilate(img, kernel, iterations=1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    #img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    img=cvtcolor_bgr_rgb(img)
    #plt.imshow(img)
    #plt.show()
    
    return (img, tuple((middle_sectionx-150, middle_sectiony-500)))
def threshold_dots3_slack(imgarray):
    img = imgarray
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    outrangemask=cv2.inRange(hsv, np.array([12,0,0]),np.array([90,255,255]))

    #outrangemask=cv2.inRange(hsv, np.array([11,255,0]),np.array([80,255,160]))
    inrangemask=cv2.bitwise_not(outrangemask)
    #hsv1=cv2.bitwise_and(hsv,hsv, mask=inrangemask)
    #inrangemask = cv2.bilateralFilter(inrangemask,9,75,75)
    #inrangemask=ndimage.filters.minimum_filter(inrangemask, (2,2))
    
    #dev,inrangemask=pcv.fill(inrangemask, inrangemask, 700,0)
    
    #img = ndimage.maximum_filter(img, size=100)
    hsv=cv2.bitwise_and(hsv, hsv, mask=inrangemask)

    inrangemask2=cv2.inRange(hsv,np.array([0,0,50]), np.array([255,255,100]))
    inrangemask2=cv2.bitwise_not(inrangemask2)
    #inrangemask2 = cv2.bilateralFilter(inraemask2,9,75,75)
    hsv=cv2.bitwise_and(hsv,hsv, mask=inrangemask2)
    inrangemask3=cv2.inRange(hsv,np.array([0,50,0]), np.array([255,255,255]))
    #inrangemask3=cv2.bitwise_not(inrangemask3)
    hsv=cv2.bitwise_and(hsv,hsv,mask=inrangemask3)
    
    img=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #img = cv2.bilateralFilter(img,5,200,200)

    img = cv2.medianBlur(img,5)
    middle_sectionx=int(img.shape[0]/2)
    middle_sectiony=int(img.shape[1]/2)
    img=img[(middle_sectionx-150):(middle_sectionx+150), (middle_sectiony-500):(middle_sectiony+500)]
    #Dilation 5x5 kernel
    
    #kernel = np.ones((3,3),np.uint8)
    #img=cv2.dilate(img, kernel)
    #img=cv2.dilate(img, kernel, iterations=1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    #img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    img=cvtcolor_bgr_rgb(img)
    #plt.imshow(img)
    #plt.show()
    
    return (img, tuple((middle_sectionx-150, middle_sectiony-500)))
def threshold_dots(imgarray):
    img = imgarray
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inrangemask=cv2.inRange(hsv, np.array([0,0,100]),np.array([179,255,255]))

    #outrangemask=cv2.inRange(hsv, np.array([11,255,0]),np.array([80,255,160]))
    #inrangemask=cv2.bitwise_not(outrangemask)
    #inrangemask = cv2.bilateralFilter(inrangemask,9,75,75)
    inrangemask=ndimage.filters.minimum_filter(inrangemask, (2,2))
    
    #dev,inrangemask=pcv.fill(inrangemask, inrangemask, 300,0)
    
    #img = ndimage.maximum_filter(img, size=100)
    hsv=cv2.bitwise_and(hsv, hsv, mask=inrangemask)
    img=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    #img = cv2.medianBlur(img,5)
    middle_sectionx=int(img.shape[0]/2)
    middle_sectiony=int(img.shape[1]/2)
    img=img[(middle_sectionx-100):(middle_sectionx+100), (middle_sectiony-500):(middle_sectiony+500)]
    #Dilation 5x5 kernel
    
    kernel = np.ones((3,3),np.uint8)
    img=cv2.dilate(img, kernel, iterations=1)
    img=cvtcolor_bgr_rgb(img)
    #plt.imshow(img)
    #plt.show()
    
    return img
def crop_out_black(imagepath):

    img = cv2.imread(imagepath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    #print(cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE))
    img34,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return crop
def crop_out_blackv2(image):

    img = image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    #print(cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE))
    img34,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return crop
def threshold_dots_withcenter3(imgarray):
    img = imgarray
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    outrangemask=cv2.inRange(hsv, np.array([30,0,0]),np.array([60,255,255]))

    #outrangemask=cv2.inRange(hsv, np.array([11,255,0]),np.array([80,255,160]))
    inrangemask=cv2.bitwise_not(outrangemask)
    #hsv1=cv2.bitwise_and(hsv,hsv, mask=inrangemask)
    #inrangemask = cv2.bilateralFilter(inrangemask,9,75,75)
    inrangemask=ndimage.filters.minimum_filter(inrangemask, (2,2))
    
    dev,inrangemask=pcv.fill(inrangemask, inrangemask, 300,0)
    
    #img = ndimage.maximum_filter(img, size=100)
    hsv=cv2.bitwise_and(hsv, hsv, mask=inrangemask)

    inrangemask2=cv2.inRange(hsv,np.array([0,0,50]), np.array([255,255,255]))
    #inrangemask2 = cv2.bilateralFilter(inrangemask2,9,75,75)
    hsv=cv2.bitwise_and(hsv,hsv, mask=inrangemask2)
    
    img=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = cv2.bilateralFilter(img,5,200,200)

    img = cv2.medianBlur(img,5)
    middle_sectionx=int(img.shape[0]/2)
    middle_sectiony=int(img.shape[1]/2)
    img=img[(middle_sectionx-200):(middle_sectionx+200), (middle_sectiony-500):(middle_sectiony+500)]
    #Dilation 5x5 kernel
    
    #kernel = np.ones((3,3),np.uint8)
    #img=cv2.dilate(img, kernel, iterations=1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    #img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    img=cvtcolor_bgr_rgb(img)
    #plt.imshow(img)
    #plt.show()
    
    return img,(middle_sectionx-200, middle_sectiony-500)
def threshold_dots_withcenter(imgarray):
    img = imgarray
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    outrangemask=cv2.inRange(hsv, np.array([10,0,0]),np.array([80,255,255]))

    #outrangemask=cv2.inRange(hsv, np.array([11,255,0]),np.array([80,255,160]))
    inrangemask=cv2.bitwise_not(outrangemask)
    #hsv1=cv2.bitwise_and(hsv,hsv, mask=inrangemask)
    #inrangemask = cv2.bilateralFilter(inrangemask,9,75,75)
    #inrangemask=ndimage.filters.minimum_filter(inrangemask, (2,2))
    
    dev,inrangemask=pcv.fill(inrangemask, inrangemask, 300,0)
    
    #img = ndimage.maximum_filter(img, size=100)
    hsv=cv2.bitwise_and(hsv, hsv, mask=inrangemask)

    inrangemask2=cv2.inRange(hsv,np.array([0,0,100]), np.array([255,255,255]))
    #inrangemask2 = cv2.bilateralFilter(inrangemask2,9,75,75)
    hsv=cv2.bitwise_and(hsv,hsv, mask=inrangemask2)
    
    img=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = cv2.bilateralFilter(img,5,200,200)

    img = cv2.medianBlur(img,5)
    middle_sectionx=int(img.shape[0]/2)
    middle_sectiony=int(img.shape[1]/2)
    img=img[(middle_sectionx-100):(middle_sectionx+100), (middle_sectiony-500):(middle_sectiony+500)]
    #Dilation 5x5 kernel
    
    #kernel = np.ones((3,3),np.uint8)
    #img=cv2.dilate(img, kernel, iterations=1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    #img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    img=cvtcolor_bgr_rgb(img)
    #plt.imshow(img)
    #plt.show()
    #plt.imshow(img)
    #plt.show()
    return [img, tuple([(middle_sectionx-100), (middle_sectiony-500)])]



    
if __name__=='__main__':
    c=NoiseRemoval.NoiseRemoval()
    #plt.imshow(cv2.imread("/Users/gghosal/Documents/GitHub/PlantTrayMeasuringPipeline/Figure_1.jpg"))
    #plt.show()
    plt.imshow(threshold_dots3(cv2.imread("/Users/gghosal/Desktop/gaurav/Plan/PlantCVCroppedTP1/101_2.jpg")))
    plt.show()
