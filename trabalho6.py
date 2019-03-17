# Processamento Digital de Imagem
# Aluno: Samuel Amico
# Matricula: 20180010181
# Trabalho 6 - Filtros

import numpy as np
import cv2
import time

capture = cv2.VideoCapture(0)


while capture.isOpened():
	ret,img = capture.read()	
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	kernel_media = np.ones((5,5),np.float32)/25
	gauss = [1,2,1,2,4,2,1,2,1]
	sobel_x = [-1,0,1,-2,0,2,-1,0,1]
	sobel_y = [-1,-2,-1,0,0,0,1,2,1]
	lapace = [0,-1,0,-1,4,-1,0,-1,0]
	
	pressed_key = cv2.waitKey(1) & 0xFF 
	# Convolve:
	print('--- Menu ----\n')
	print('a- media ----\n')
	print('b- gauss ----\n')
	print('c- horizontal ----\n')
	print('d- vertical ----\n')
	print('e- laplace ----\n')
	print('f- laplacegauss ----\n')
	print('z- Sair ----\n')
	#a)
	if pressed_key == ord("a"):
		dst = cv2.filter2D(img,-1,kernel_media)
		cv2.imshow(" media ",dst)
		#cv2.imwrite('FiltroMedia.png',dst)
		
	elif pressed_key == ord("b"):
		dst = cv2.GaussianBlur(img,(11,11),0)
		cv2.imshow(" gauss ",dst)
		#cv2.imwrite('FiltroGauss.png',dst)
		
	elif pressed_key == ord("c"):
		dst = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
		cv2.imshow(" Sobel-x ",dst)
		#cv2.imwrite('FiltroSx.png',dst)
			
	elif pressed_key == ord("d"):
		dst = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
		cv2.imshow(" Sobel-y ",dst)
		#cv2.imwrite('FiltroSy.png',dst)
		
	elif pressed_key == ord("e"):
		dst = cv2.Laplacian(img,cv2.CV_64F,ksize=1)
		cv2.imshow(" Laplace ",dst)
		#cv2.imwrite('FiltroLaplace.png',dst)
				
	elif pressed_key == ord("f"):
		dst1 = cv2.GaussianBlur(img,(5,5),0)
		dst = cv2.Laplacian(dst1,cv2.CV_64F,ksize=1)
		cv2.imshow(" LaplaGauss ",dst)
		cv2.imwrite('FiltroLaplaceGauss.png',dst)
						
	if pressed_key == ord("z"):
		cv2.imwrite('ImgOrig.png',img)
		break
		
	cv2.imshow(" Orig ",img)
cv2.destroyAllWindows()
capture.release() 		

