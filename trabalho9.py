import numpy as np
import cv2
import math
import time
import random

def nothing(x):
	pass

STEP = 5
####### TRACKBAR #########
cv2.namedWindow('bar')
cv2.createTrackbar('top','bar',0,255,nothing)

ab = 10
bl = 3*ab

image = cv2.imread('golden.jpg',0)
frame = cv2.imread('golden.jpg')
height, width = image.shape
print("height - y: ",height,"width - x: ",width)

xranges = [0]*height
yranges = [0]*width

for i in range(height):
	xranges[i] = xranges[i]*STEP+STEP/2
for i in range(width):
	yranges[i] = yranges[i]*STEP+STEP/2
	
xranges.sort()


points = np.ones((height,width,3))*255
points = points.astype(np.uint8)

# Algoritimo:

while True:
	pressed_key = cv2.waitKey(1) & 0xFF 
	ab = cv2.getTrackbarPos('top','bar')
	canny = cv2.Canny(image, ab, bl)
	cv2.imshow('bar', canny)
	if pressed_key == ord("z"):
		break


for i in range(len(yranges)):
	yranges.sort()
	for j in range(len(xranges)):
		if(canny[j,i] == 255):
			x = int(i+(random.random() % (4)) - 2)
			y = int(j+(random.random() % (4)) - 2)
			color_b = int(frame[y,x,0])
			color_g = int(frame[y,x,1])
			color_r = int(frame[y,x,2])
			#print("x,y=",x,y)
			cv2.circle(points,(x,y),4,(color_b,color_g,color_r),-1)
		else:
			x = int(i+(random.random()%(4))- 1)
			y = int(j+(random.random()%(4))- 1)
			color_b = int(frame[y,x,0])
			color_g = int(frame[y,x,1])
			color_r = int(frame[y,x,2])
			#print("x,y = ",x,y)
			#print("b,g,r = ",color_b,color_g,color_r)
			cv2.circle(points,(x,y),3,(color_b,color_g,color_r),-1)
			

cv2.imshow('points', points)	


cv2.waitKey(0)
#cv2.imwrite('canny.png',canny)
#cv2.imwrite('golden.png',golden)
cv2.imwrite('points.png',points)
		
cv2.destroyAllWindows()
