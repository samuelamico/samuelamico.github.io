# Digital Image Processing
# Student: Samuel Amico
# Number: 20180010181
# Exercise 2.2 - bolhas.png


import numpy as np
import cv2
import time


image = cv2.imread('bolhas.png')


height, width, ch = image.shape
print("height - y: ",height,"width - x: ",width)

# Color:
cor_B = 255
cor_G = 0
cor_R = 0

# P1 = top-left & P2 = bottom-right

P1x = input("Ponto 1 x - top:")
P1y = input("Ponto 1 y - top:")
P2x = input("Ponto 2 x - bot:")
P2y = input("Ponto 2 y - bot:")
print("P1 = (",P1x,",",P1y,")  ","P2 = (",P2x,",",P2y,")")


if (image is not None):
	cv2.imshow("Original", image)

k = cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.rectangle(image,(int(P1x-3),int(P1y-3)),(int(P2x+3),int(P2y+3)),(cor_B,cor_G,cor_R),2)
cv2.imshow("Rec in Image", image)
#cv2.imwrite('RecBolha2.png',image)
k = cv2.waitKey(0)

# ROI --> Color
for i in range(P1x,P2x):
	for j in range(P1y,P2y):
		image[i,j] = [cor_B,cor_G,cor_R]

cv2.imshow("ROI Colored", image)
k = cv2.waitKey(0)
#cv2.imwrite('BolhaCor.png',image)
cv2.destroyAllWindows()
