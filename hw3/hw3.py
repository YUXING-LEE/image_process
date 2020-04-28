import cv2
import numpy as np

def toGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image, thresholdValue, maxValue, thresholdType):
    return cv2.threshold(image, thresholdValue, maxValue, thresholdType)[1]

def gaussianBlur(image, kernelSize):
    return cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)

def canny(image, kernelSize, minThreshold, maxThreshold):
    return cv2.Canny(image, minThreshold, maxThreshold)

def dilate(image, kernelSize, iterations):
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    return cv2.dilate(image, morph_kernel, iterations=iterations)

folder = "origin/"
road = folder + "road4"
src = cv2.imread(road +'.jpeg')
src = gaussianBlur(src, 3)
graySrc = toGray(src)
rows,cols  =  src.shape[:2]
kernel = np.array( [[1, 2, 4],
					[128, 0, 8],
					[64, 32, 16]])
										
#kernelSize = 3
#radius = int(kernelSize / 2)
#result = np.zeros((rows,cols),np.uint8)
#for row in range(radius,rows-radius):
#	for col in range(radius,cols-radius):
#		value = np.zeros((kernelSize,kernelSize),np.uint8)
#		Temp = graySrc[row][col]
#		array = graySrc[row-radius:row+radius+1,col-radius:col+radius+1]
#		diff = array - Temp
#		value[diff > 0] = 1
#		value = value * kernel
#		sum_value = np.sum(value)
#		result[row][col] = sum_value

aveList, stdDevList = cv2.meanStdDev(graySrc)
ave = aveList[0][0]
stdDev = stdDevList[0][0]
o1 = np.where((graySrc < ave - stdDev), 255, 0).astype(np.uint8)
o2 = np.where((graySrc > ave - stdDev) & (graySrc < ave), 255, 0).astype(np.uint8)
o3 = np.where((graySrc > ave) & (graySrc < ave + stdDev), 255, 0).astype(np.uint8)
o4 = np.where((graySrc > ave + stdDev), 255, 0).astype(np.uint8)

result = np.zeros((rows,cols, 3),np.uint8)
result[o1 == 255] = (255, 0, 0)
result[o2 == 255] = (0, 255, 0)
result[o3 == 255] = (0, 0, 255)
result[o4 == 255] = (255, 255, 0)

cv2.imshow("o1", o1)
cv2.imshow("o2", o2)
cv2.imshow("o3", o3)
cv2.imshow("o4", o4)
cv2.imshow("result", result)
cv2.waitKey(0)
		
		
