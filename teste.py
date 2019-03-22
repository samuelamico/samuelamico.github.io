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

# Utilizando a equacao para criar a imagem preto e branco:


image = cv2.imread('suica.jpg')
image = cv2.resize(image,None,fx=0.4,fy=0.4,interpolation=cv2.INTER_CUBIC)

height, width,ch = image.shape
print("height - y: ",height,"width - x: ",width,ch)

# Imagem borrada:
imageA = image.copy()
for i in range(4):
	imageA = cv2.GaussianBlur(imageA,(3,3),0)


# Imagem Normal:
imageB = image.copy()


image_1 = np.ones((height,width,3))*255
image_2 = np.ones((height,width,3))*255
image_3 = np.ones((height,width,3))

ab = 33.0
bl = 3.4
cen = 40.0

# valores para usar: ab =22.0, bl=2.0, cen=80.0


for i in range(height):
	x = i*100.0/width
	val = (math.tanh((x-ab)/bl) - math.tanh((x-cen)/bl) )/2.0
	pixel_val = 255.0 * val
	for j in range(width):
		image_1[i,j] = pixel_val
		image_2[i,j] = 255.0 - pixel_val
		 
cv2.imshow("bar",image_1)
image_1 = cv2.addWeighted(image_1,1.0,image_1,1.0,0,dtype=cv2.CV_8U)

vis1 = cv2.multiply(image_2,image_1,dtype=cv2.CV_8U)
vis = cv2.multiply(imageB,image_1,dtype=cv2.CV_8U)
vis3 = cv2.addWeighted(imageB,1.0,cv2.bitwise_not(vis1),0.3,0,dtype=cv2.CV_8U)



teste = cv2.addWeighted(vis,1.0,vis,1.0,0,dtype=cv2.CV_8U)
#cv2.imshow("Imagem blurring perto",teste)

## Nao estao UINT8
vis1 = cv2.multiply(image_2,image_1,dtype=cv2.CV_8U)
vis2 = cv2.addWeighted(imageB,1.0,cv2.bitwise_not(vis1),1.0,0,dtype=cv2.CV_8U)
vis3 = cv2.addWeighted(imageA,1.0,vis1,1.0,0,dtype=cv2.CV_8U)
#cv2.imshow("Imagem Central",vis2)
#cv2.imshow("Imagem blurring ",vis3)

## COnvertendo --- apartir daqui ja foram convertidas as imagens para 8UINT
image_3 = vis1.astype(np.uint8)
image_4 = image_1.astype(np.uint8)

ret, thr = cv2.threshold(image_3,10,255,cv2.THRESH_BINARY)
ret1, thr1 = cv2.threshold(image_4,10,255,cv2.THRESH_BINARY)
	

vis22 = cv2.addWeighted(imageB,1.0,cv2.bitwise_not(thr),1.0,0,dtype=cv2.CV_8U)
vis33 = cv2.addWeighted(imageA,1.0,thr,1.0,0,dtype=cv2.CV_8U)


cv2.imshow("Imagem Central",vis22)
cv2.imshow("Imagem blurring ",vis33)
#cv2.imshow("Imagem blurring de perto ",teste)

fin = cv2.bitwise_and(vis22,vis33)
cv2.imshow("fin",fin)

k = cv2.waitKey(0)
cv2.destroyAllWindows()


