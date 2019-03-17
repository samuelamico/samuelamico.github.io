# Processamento Digital de Imagem
# Aluno: Samuel Amico
# Matricula: 20180010181
# Trabalho 5 - Histogramas



#### Histogram Comparationz
import numpy as np
import cv2
import time


global flag
flag = True
drawing = False
img = None
c = 1
boxes = []
capture = cv2.VideoCapture(0)
_,img = capture.read()

# a funcao aki chama a interupcao do mouse
# o intuito e voce fazer uma area retangular para ser seu roi    
def on_mouse(event,x,y,flags,params):
	global boxes,drawing
	if event == cv2.EVENT_LBUTTONDOWN:
		print('Start Mouse Position:' ,str(x),str(y))
		s = (x,y)
		boxes.append(s)
		drawing = True
		cv2.circle(img,(x,y),4,(0,255,0),2)	
		cv2.imshow('img',img)
	elif event == cv2.EVENT_LBUTTONUP:
		print('END Mouse Position:',str(x),str(y))
		e = (x,y)
		boxes.append(e)
		drawing = False	
		mode = True
		
def ROI_segment(retangulo,capturing): 	
	hsv_roi = cv2.cvtColor(retangulo, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
	roi_hist = cv2.calcHist([hsv_roi], [0],None,[16],[0,180])
	return cv2.normalize(roi_hist,roi_hist, 0, 255, cv2.NORM_MINMAX) 		
		
while capture.isOpened():
	cv2.namedWindow('img')
	#CHAMA A FUNC MOUSE ALI EM CIMA
	cv2.setMouseCallback('img',on_mouse,0)		
	ret,img = capture.read(0)	
	pressed_key = cv2.waitKey(1) & 0xFF 
	if not ret:
		break
	# enquanto esta voce nao escolheu os 4 pontos
	# ele vai ficar nesse if
	if len(boxes)<4:
		orig = img.copy()
		#copia o primeiro frame da imagem
		print("entrou no roi")
		# fica no while ate acabar os 4 pontos, a imagem nao se atualiza
		while len(boxes) < 4:
			cv2.imshow('img',img)
			cv2.waitKey(0)	
		boxes = np.array(boxes)
		s = boxes.sum(axis = 1)
		# ele vai pegar os menores pontos
		t1 = boxes[np.argmin(s)]
		# pegar os maiores pontos
		br = boxes[np.argmax(s)]
		# separa a regiao desses pontos
		roi = orig[t1[1]:br[1],t1[0]:br[0]]
		roi_area = (t1[0],t1[1],br[0],br[1])
		# chama aquela funcao que pega o histograma desse retangulo que voce selecionou
		roi_hist = ROI_segment(roi,img)		
		flag = False
	if len(boxes)>4:
		ret,img = capture.read(0)
		# cria o hist do novo frame
		#new_orig = img.copy()
		new_roi = img[t1[1]:br[1],t1[0]:br[0]]
		new_hsv = cv2.cvtColor(new_roi, cv2.COLOR_BGR2HSV)
		cv2.rectangle(img,(t1[0],br[0]),(t1[1],br[1]),(0,255,0),3)
		#cv2.imshow('hsv_roi_new',new_roi)
		#cv2.imshow('roi',roi)
		
		base_hist = cv2.calcHist([new_hsv], [0],None,[16],[0,180])
		base_roi = cv2.normalize(base_hist,base_hist, 0, 255, cv2.NORM_MINMAX) 		
		# comparamos agora os dois ROI
		
		number = cv2.compareHist( base_roi, roi_hist, cv2.HISTCMP_CORREL);
		print("number comparation=",number)

		#pts = np.int0(cv2.boxPoints(ret1))
		#cv2.polylines(img,[pts],True,255,2)
		#cv2.imshow("Final",img)	

	cv2.imshow('img',img)
	if pressed_key == ord("z"):
		cv2.imwrite('Motion.png',img)
		break
cv2.destroyAllWindows()
capture.release() 
	
