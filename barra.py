# Digital Image Processing
# Student: Samuel Amico
# Number: 20180010181
# Tilt shift Image


import numpy as np
import cv2
import math
import time

def nothing(x):
	pass

####### TRACKBAR #########
cv2.namedWindow('bar')
cv2.createTrackbar('abertura','bar',0,100,nothing)
cv2.createTrackbar('Blur','bar',0,100,nothing)
cv2.createTrackbar('Center','bar',0,100,nothing)

##########################

image = cv2.imread('suica.jpg')
image = cv2.resize(image,None,fx=0.4,fy=0.4,interpolation=cv2.INTER_CUBIC)

height, width,ch = image.shape
print("height - y: ",height,"width - x: ",width,ch)


# Matrizes de filtro
kernel_media = np.ones((5,5),np.float32)/25
mask = np.zeros((3,3),np.float32)/9
kernel2 = np.array([[0,0,0],[1,1,1],[0,0,0]])

#cv2.imshow("Image", image)

# Imagem borrada:
imageA = image.copy()
for i in range(4):
	imageA = cv2.GaussianBlur(imageA,(3,3),0)
#cv2.imshow("ImageA", imageA)

# Imagem Normal:
imageB = image.copy()


# Utilizando a equacao para criar a imagem preto e branco:

image_1 = np.ones((height,width,3))*255
image_2 = np.ones((height,width,3))*255
#cv2.imshow("image_1",image_1)
'''
ab =cv2.getTrackbarPos('abertura','bar')
bl =cv2.getTrackbarPos('Blur','bar')
cen =cv2.getTrackbarPos('aCenter','bar')

vis = cv2.multiply(image_2,image_1)
cv2.imshow("bar",vis)
'''


def barra(ab,bl,cen):
	for i in range(height):
		x = i*100.0/width
		bl = (bl*0.1)+0.1
		val = (math.tanh((x-ab)/bl) - math.tanh((x-cen)/bl) )/2
		pixel_val = 255*val
		for j in range(width):
			image_1[i,j] = pixel_val
			image_2[i,j] = 255 - pixel_val


# Merge images:
#vis = cv2.multiply(image_2,image_1)
#cv2.imshow("bar",vis)
#k = cv2.waitKey(0)
#cv2.destroyAllWindows()



while (True):
	pressed_key = cv2.waitKey(1) & 0xFF 
	ab =cv2.getTrackbarPos('abertura','bar')
	bl =cv2.getTrackbarPos('Blur','bar')
	cen =cv2.getTrackbarPos('aCenter','bar')
	# chama func
	barra(ab,bl,cen)
	
	vis = cv2.multiply(image_2,image_1)
	cv2.imshow("bar",vis)
	#teste = np.ones((height,width,3))*255
	teste = cv2.addWeighted(imageA,0.7,vis,0.3,0,dtype=cv2.CV_8U)
	teste1 = cv2.addWeighted(imageB,0.7,cv2.bitwise_not(vis),0.3,0,dtype=cv2.CV_8U)
	#cv2.imshow("image 1",teste)
	#cv2.imshow("image 2",teste1)

	final = cv2.addWeighted(teste1,1,imageA,0.8,0,dtype=cv2.CV_8U)
	#cv2.imwrite("TitlShift.png",final)
	
	
	if pressed_key == ord("z"):
		#cv2.imwrite('ImgOrig.png',img)
		cv2.imshow("final",final)
		break

k = cv2.waitKey(0)
cv2.destroyAllWindows()


























