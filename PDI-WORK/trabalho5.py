# Processamento Digital de Imagem
# Aluno: Samuel Amico
# Matricula: 20180010181
# Trabalho 5 - Histogramas

import numpy as np
import cv2
import time
import scipy.ndimage as ndimage

capture = cv2.VideoCapture(0)

# captura uma imagem antiga - old
ret_old,img_old = capture.read()

b,g,r= cv2.split(img_old)
# mude img_old por bgr_planes para testar depois

hist_size = 256
hist_old,bins_old = cv2.calcHist(b,0,None,256,[0,256]) # Blue channel	
height, width = img_old.shape[:2]

hist_w = width
hist_h = height
bin_w = int(round(hist_w/hist_size))
histImage = np.zeros((hist_h,hist_w,3),dtype=np.uint8)

cv2.normalize(hist_old,hist_old,alpha=0,beta=hist_h, norm_type=cv.NORM_MINMAX)

histB = old_hist.copy()

while capture.isOpened():
	ret,img = capture.read()	
	#old_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# dimensoes da imagem
	height, width = image.shape[:2]
	
	bgr_planes = cv2.split(img)
	hist_size = 256
	hist,bins = cv2.calcHist(img.flatten(),[0],None,[256],[0,256]) # Blue channel	
	# Normalizando:
	hist_w = width
	hist_h = height
	bin_w = int(round(hist_w/hist_size))
	histImage = np.zeros((hist_h,hist_w,3),dtype=np.uint8)
	cv2.normalize(hist,hist,alpha=0,beta=hist_h, norm_type=cv.NORM_MINMAX)
	# Comparando Histogramas:
	histor = cv2.compareHist(histImage,histB,CV_COMP_CORREL)
	if(histor < 0.01):
		print("Motion detect")
		
	for i in range(1,hist_size):
		cv2.line(histImage,(bin_w*(i-1),hist_h - int(round(hist[i-1])) ),(bin_w*(i),hist_h - int(round(hist[i])) ), (255,0,0), thickness=2)
	cv2.imshow('image',img)
	cv2.imshow('calc Hist',histImage)
	
	pressed_key = cv2.waitKey(1) & 0xFF
	if pressed_key == ord("z"):
		#cv2.imwrite('HistEqualizado.png',res)
		break
	
