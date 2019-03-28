import numpy as np
import cv2
import math
import time

capture = cv2.VideoCapture('walking.avi')

ret1, frame1 = capture.read()
frame1 = cv2.resize(frame1, None,fx=1.1, fy=1.1, interpolation = cv2.INTER_LINEAR)
height, width,ch = frame1.shape
print("height - y: ",height,"width - x: ",width,ch)

image_1 = np.ones((height,width,3))*255
image_2 = np.ones((height,width,3))*255
image_3 = np.ones((height,width,3))
	
ab = 33.0
bl = 3.4
cen = 40.0

for i in range(height):
	x = i*100.0/width
	val = (math.tanh((x-ab)/bl) - math.tanh((x-cen)/bl) )/2.0
	pixel_val = 255.0 * val
	for j in range(width):
		image_1[i,j] = pixel_val
		image_2[i,j] = 255.0 - pixel_val

image_1 = cv2.addWeighted(image_1,1.0,image_1,1.0,0,dtype=cv2.CV_8U)
vis1 = cv2.multiply(image_2,image_1,dtype=cv2.CV_8U)
## COnvertendo --- apartir daqui ja foram convertidas as imagens para 8UINT
image_3 = vis1.astype(np.uint8)
ret, thr = cv2.threshold(image_3,10,255,cv2.THRESH_BINARY)

### Salvando video:
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc,10.0,(640,480))


while capture.isOpened():
	time.sleep(.05)
	ret, frame = capture.read()
	frame = cv2.resize(frame, None,fx=1.1, fy=1.1, interpolation = cv2.INTER_LINEAR)
	imageA = frame.copy()
	imageB = frame.copy()
	for i in range(5):
		imageA = cv2.GaussianBlur(imageA,(3,3),0)

	vis22 = cv2.addWeighted(imageB,1.0,cv2.bitwise_not(thr),1.0,0,dtype=cv2.CV_8U)
	vis33 = cv2.addWeighted(imageA,1.0,thr,1.0,0,dtype=cv2.CV_8U)
	fin = cv2.bitwise_and(vis22,vis33)
	
	#cv2.imshow('Pedestrian Detection', frame)
	cv2.imshow('Pedestrian Tilt', fin)
	#out.write(fin)
	c = cv2.waitKey(1)
	if c == ord("z"):
		break

# Close the capturing device
capture.release()
#out.release()
# Close all windows
cv2.destroyAllWindows()
	
	
