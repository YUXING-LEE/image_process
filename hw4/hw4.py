import numpy as np
import cv2
import random

img = cv2.imread('road3.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_TRIANGLE )

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
cv2.imshow("sure_fg",sure_fg)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

target = np.zeros(img.shape,np.uint8)
labels = np.unique(markers)
for label in labels:
	if label == -1:
		continue
	target[markers == label] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

cv2.imshow("As",img)
a = cv2.add(img,target)
cv2.imshow("ASS",a)
cv2.imshow("Ans",target)
#cv2.imwrite("result_0" +"road3" +".jpeg",sure_fg)
#cv2.imwrite("result_1" +"road3" +".jpeg",img)
#cv2.imwrite("result_2" +"road3" +".jpeg",target)
#cv2.imwrite("result_3" +"road3" +".jpeg",a)
cv2.waitKey(0)
