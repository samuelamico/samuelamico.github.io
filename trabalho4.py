# Processamento Digital de Imagem
# Aluno: Samuel Amico
# Matricula: 20180010181
# Trabalho 3 - referente a imagem bolhas.png

import numpy as np
import cv2
import time

capture = cv2.VideoCapture(0)


while capture.isOpened():
	ret,img = capture.read()	
	old_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	equ = cv2.equalizeHist(old_gray)
	res = np.hstack((old_gray,equ))
	
	#cv2.imshow('image',old_gray)
	cv2.imshow('Histogram equ',res)
	pressed_key = cv2.waitKey(1) & 0xFF
	if pressed_key == ord("z"):
		break
cv2.destroyAllWindows()
capture.release() 		
