# Processamento Digital de Imagem
# Aluno: Samuel Amico
# Matricula: 20180010181
# Trabalho 3 - referente a imagem bolhas.png

import numpy as np
import cv2
import time
import scipy.ndimage as ndimage

# Ler imagem grayscale:
image = cv2.imread('bolhas.png',0)
cv2.imshow(' Original Image',image)
k = cv2.waitKey(0)

# dimensoes da imagem
height, width = image.shape[:2]



### apagando as bolhas que tocam as paredes:
bolhas_parede = 0
img = image
for i in range(height):
	if(img[i,0]==255):
		cv2.floodFill(img,None,(0,i),(0,),(0,),(0,))
	if(img[i,width-1]==255):
		cv2.floodFill(img,None,(width-1,i),(0,),(0,),(0,))

for i in range(width):
	if(img[0,i]==255):
		cv2.floodFill(img,None,(i,0),(0,),(0,),(0,))
	if(img[height-1,i]==255):
		cv2.floodFill(img,None,(i,height-1),(0,),(0,),(0,))		

cv2.imshow("Image refresh",image)
k = cv2.waitKey(0)
#cv2.imwrite('BolhaAtualizada.png',image)
cv2.destroyAllWindows()

## Find Holes and count holes too
image_fill = image.copy()

obj=0
mask = np.zeros((height+2,width+2),np.uint8)


labels, n_regions = ndimage.label(image)
print("Numbers of bublles = ", n_regions)

## Count holes:
for i in range(height):
	for j in range(width):
		if(image[i,j]==0):
			obj=obj+1
			cv2.floodFill(image_fill,mask,(j,i),obj)

cv2.imshow("Floodfill",image_fill)
k = cv2.waitKey(0)
#cv2.imwrite('BolhaCheia.png',image_fill)

for i in range(height):
	for j in range(width):
		if(image[i,j]==255):
			obj=obj+1
			c=cv2.floodFill(image_fill,mask,(j,i),0)

cv2.imshow("Holes",image_fill)
k = cv2.waitKey(0)
#cv2.imwrite('BolhaFuro.png',image_fill)
cv2.destroyAllWindows()

labels, n_regions = ndimage.label(image_fill)
print("Numbers of holes = ", n_regions)

	
			
## Resultado		

cv2.destroyAllWindows()














