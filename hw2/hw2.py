import cv2
import numpy as np

def  canny(img):
    edges = cv2.Canny(img,100,200)
    return edges

def sobel(img):
    gray = cv2.GaussianBlur(img,(3,3),0)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)  
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    cv2.imwrite("sobel_x.png", sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    cv2.imwrite("sobel_y.png", sobely)
    sobel = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    return sobel

src = cv2.imread('test.jpg',1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.png", gray)
canny_img = canny(gray)
cv2.imwrite("canny.png", canny_img)
sobel_img = sobel(gray)
cv2.imwrite("sobel.png", sobel_img)
