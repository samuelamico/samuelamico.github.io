
import numpy as np
import cv2
import math
import time

pressed_key = cv2.waitKey(1) & 0xFF 


# Utilizando a equacao para criar a imagem preto e branco:


image = cv2.imread('suica.jpg')
image = cv2.resize(image,None,fx=0.4,fy=0.4,interpolation=cv2.INTER_CUBIC)

height, width,ch = image.shape
print("height - y: ",height,"width - x: ",width,ch)

# Imagem borrada:
imageA = image.copy()
for i in range(4):
	imageA = cv2.GaussianBlur(imageA,(3,3),0)
#cv2.imshow("ImageA", imageA)

# Imagem Normal:
imageB = image.copy()


image_1 = np.ones((height,width,3))*255
image_2 = np.ones((height,width,3))*255

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


cv2.imshow("image1",image_1)
image_1 = cv2.addWeighted(image_1,1.0,image_1,1.0,0,dtype=cv2.CV_8U)

vis = cv2.multiply(imageB,image_1,dtype=cv2.CV_8U)
teste = cv2.addWeighted(vis,1.0,vis,1.0,0,dtype=cv2.CV_8U)
cv2.imshow("Imagem blurring perto",teste)


vis1 = cv2.multiply(image_2,image_1,dtype=cv2.CV_8U)
vis2 = cv2.addWeighted(imageB,1.0,cv2.bitwise_not(vis1),1.0,0,dtype=cv2.CV_8U)
vis3 = cv2.addWeighted(imageA,1.0,vis1,1.0,0,dtype=cv2.CV_8U)
cv2.imshow("Imagem Central",vis2)
cv2.imshow("Imagem blurring ",vis3)



k = cv2.waitKey(0)

if pressed_key == ord("z"):
	#cv2.imwrite('ImgOrig.png',img)
	#cv2.imwrite("TitlShift.png",final)
	cv2.destroyAllWindows()

