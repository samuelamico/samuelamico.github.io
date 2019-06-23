import cv2
import numpy as np 
import dlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imutils import face_utils

img = mpimg.imread('face1.jpg')

resize = cv2.resize(img, (100, 200), interpolation = cv2.INTER_LINEAR) 


im = np.float32(img) / 255.0

# Vamos calcular o gradiente
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

face_detect = dlib.get_frontal_face_detector()

rects = face_detect(img, 1)

x_maior = 0
y_maior = 0
w_maior = 0
h_maior = 0

for (i, rect) in enumerate(rects):
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	if(w > w_maior):
		x_maior = x
		y_maior = y
		w_maior = w
		h_maior = h
	
	print(x,y,w,h)
	
	cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)

print(x_maior,y_maior,w_maior,h_maior)
img_final = img[y_maior:y_maior+h_maior,x_maior:x_maior+w_maior]

cv2.imwrite('new_face.jpg',img_final)

print(img.shape[:2])
plt.figure()
plt.imshow(img_final)
plt.show()
