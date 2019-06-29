##SimpleBlobDetector
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import plantcv as pcv 
sys.path.append("/Users/gghosal/Documents/GitHub/PlantTrayMeasuringPipeline/")
import ImageProcUtil
image=cv2.imread('/Users/gghosal/Desktop/gaurav/Plan/PlantCVCroppedTP1/101_1.jpg')
p=ImageProcUtil.threshold_dots3(image)
thresholded=cv2.threshold(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
thresholded=thresholded[1]
plt.imshow(thresholded)
plt.show()


params=cv2.SimpleBlobDetector_Params()
params.filterByArea=True
params.minArea=1000
params.maxArea=4000
detector=detector=cv2.SimpleBlobDetector_create(params)
kps=detector.detect(p)
print(kps)

im_with_keypoints = cv2.drawKeypoints(p, kps, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('h',im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
