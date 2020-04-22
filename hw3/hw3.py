import cv2
import numpy as np

road = "road2"
src = cv2.imread(road +'.jpeg',0)
rows,cols  =  src.shape[:2]

kernel = np.array([[1, 2, 4],
											[128, 0, 8],
											[64, 32, 16]])
											
result = np.zeros((rows,cols),np.uint8)

for row in range(1,rows-1):
	for col in range(1,cols-1):
		value = np.zeros((3,3),np.uint8)
		Temp = src[row][col]
		array = src[row-1:row+2,col-1:col+2]
		diff = array - Temp
		value[diff > 0] = 1
		value = value*kernel
		sum_value = np.sum(value)
		result[row][col] = sum_value
	print(row)

cv2.imshow("result",result)
cv2.imwrite("result_" +road +".jpeg",result)
cv2.waitKey(0)
		
		
