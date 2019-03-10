# Digital Image Processing
# Student: Samuel Amico
# Number: 20180010181
# Exercise 1.1 - biel.png

import numpy as np
import cv2
import time

# Negative.py :

image = cv2.imread('biel.png')
#cv2.imwrite('bielgray.png',image)

height, width, ch = image.shape
print("height - y: ",height,"width - x: ",width)


#  P1 = top-left & P2 = bottom-right
# 10,10 - 150,150
P1x = input("Ponto 1 x - top:")
P1y = input("Ponto 1 y - top:")
P2x = input("Ponto 2 x - bot:")
P2y = input("Ponto 2 y - bot:")
print("P1 = (",P1x,",",P1y,")  ","P2 = (",P2x,",",P2y,")")


if (image is not None):
	cv2.imshow("Original", image)

k = cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.rectangle(image,(int(P1x-3),int(P1y-3)),(int(P2x+3),int(P2y+3)),(0,255,0),2)
cv2.imshow("Rec inside the image", image)
k = cv2.waitKey(0)
#cv2.imwrite('RecImage.png',image)

# Apply Negative efect
for i in range(P1x,P2x):
	for j in range(P1y,P2y):
		image[i,j] = 255 - image[i,j]

cv2.imshow("Negative", image)
k = cv2.waitKey(0)
#cv2.imwrite('negativebiel.png',image)
cv2.destroyAllWindows()


# ------------------------------> y
#|(0,0)                        |
#|				               |
#|				               |
#|				               |
#|				               |
#|				               |
#|				               |
#|				               |
#|               (width,height)|
#X------------------------------
