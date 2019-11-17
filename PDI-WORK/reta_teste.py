#### Autonomos car
import numpy as np
import cv2
import time

capture = cv2.VideoCapture(1)

def rescale_frame(capturing, wpercent=50, hpercent=50):
    width = int(capturing.shape[1] * wpercent / 100)
    height = int(capturing.shape[0] * hpercent / 100)
    return cv2.resize(capturing, (width, height), interpolation=cv2.INTER_AREA)

def white_seg(img,hsv,thr):
	upper = np.array([0,0,212])
	lower = np.array([131,255,255])
	mask1 = cv2.inRange(hsv,lower,upper)
	out1 = cv2.bitwise_and(img,img,mask=mask1)
	kernel = np.ones((15,15),np.uint8)                  # destruindo os ruidos
	mask = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
	out = cv2.bitwise_and(img,img,mask=mask)
	#cv2.imshow('filter',out)
	# teste para melhorar:
	final = cv2.bitwise_and(out,out1)
	final = cv2.blur(final,(5,5))
	kernel_new = np.ones((15,15),np.uint8)
	mask_new = cv2.morphologyEx(final,cv2.MORPH_OPEN,kernel_new)
	cv2.imshow('filter',out)
	return out

def filtragem(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	ret,threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	for i in range(4):
		blurred = cv2.GaussianBlur(threshold,(11,11),0)
	white_img = white_seg(blurred,hsv,threshold)
	
	return white_img
	
def contorno(white_img,frame):
	#canny = cv2.Canny(white_img, 50, 200)
	# depois tente aplicar contorno no canny
	ret1,thr = cv2.threshold(white_img, 127, 255, cv2.THRESH_BINARY)
	result = cv2.findContours(thr,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	cont,hierarchy = result if len(result) == 2 else result[1:3]
	if len(cont) > 0:
		areas = [cv2.contourArea(c) for c in cont]
		max_index = np.argmax(areas)
		cont_max = cont[-1]
		M = cv2.moments(cont[0])
		if (M['m00'] != 0):	
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			cv2.circle(frame,(cx,cy),8,(0,255,105),3)
			return (cx,cy)
	return (0,0)


#### Funcs matematicas para o controlador:
def regressao_linear(x1,x2,x3,y1,y2,y3,height,width,frame):
	x_medio = float((x1+x2+x3)/3)
	y_medio = float((y1+y2+y3)/3)
	sum1 = x1*y1+x2*y2+x3*y3
	a1 = float(3.0*(sum1) - (x1+x2+x3)*(y1+y2+y3) )/float(3.0*(x1**2+x2**2+x3**2) - ((x1+x2+x3))**2  )
	a0 = y_medio - a1*x_medio
	# y = a0 + a1*x
	# ponto 0: 
	x1 = 0   # na real isso eh o y na func do cv2.line   
	y1 = a0  # na real isso eh o x na func do cv2.line
	# ponto F:
	xf = height
	yf = a0 + a1*xf
	cv2.line(frame,(y1,x1),(yf,xf),(0,255,105),4)
	return (y1,yf)	
	
	
	
	
while True:
	_,img = capture.read()
	pressed_key = cv2.waitKey(1) & 0xFF
	frame = rescale_frame(img)
	height,width = frame.shape[:2]
	frame_new = filtragem(frame)
	#print(width,height)
	# tracando a reta de entrada para o controlador
	# as imagens funcionam mais ou menos assim:
	# ------------------------------
	#|(0,0)                        |
	#|				               |
	#|				               |
	#|				               |
	#|				               |
	#|				               |
	#|				               |
	#|				               |
	#|               (width,height)|
	# ------------------------------
	#### copy 1:
	frame_copy = frame[0:height/3,100:width] # [coordenas y-(y0:yf), coordenas x(x0,xf)]
	frame_seg1 = filtragem(frame_copy)
	(x1,y1)=contorno(frame_seg1,frame_copy)
	print(x1,y1)
	cv2.imshow('copy',frame_copy)
	### copy 2:
	frame_copy2 = frame[height/3:2*height/3,100:width]
	frame_seg2 = filtragem(frame_copy2)
	(x2,y2)=contorno(frame_seg2,frame_copy2)
	cv2.imshow('copy2',frame_copy2)
	### copy 3:
	frame_copy3 = frame[2*height/3:height,100:width]
	frame_seg3 = filtragem(frame_copy3)
	(x3,y3)=contorno(frame_seg3,frame_copy3)
	cv2.imshow('copy3',frame_copy3)
	####### CONTROLADOR ###########
	height,width = frame.shape[:2]
	middle_x = width/2
	cv2.line(frame,(middle_x,0),(middle_x,height),(255,0,0),4)
	##### atende aqui samuel pois os valores x1,x2,x3,y1,y2,y3 estao feitos na escala
	##### do frame_copy, entao adicione 100 ao x e height/3 no y
		#cv2.circle(frame,(x1+100,y1),3,(200,255,105),3)	
		#cv2.circle(frame,(x2+100,y2+height/3),3,(200,255,105),3)
		#cv2.circle(frame,(x3+100,y3+2*height/3),3,(200,255,105),3)
	# reta ligando pontos
	x1 = int( x1 + 100 )
	x2 = int (x2 + 100)
	x3 = int(x3 + 100)
	y2 = int(y2+height/3)
	y3 = int(y2+2*height/3)	 	
	cv2.line(frame,(x3,y3),(x1,y1),(200,255,105),4)	
	'''
	# reta regressao linear dos pontos obtidos:
	(x0,xf)=regressao_linear(x1,x2,x3,y1,y2,y3,height,width,frame) # verificar func
	# U(t) = eh exatamente a posicao media do xo e xf da reta regressao
	u = (float(x0 + xf)/2.0)
	## funcao erro: e = u - middle_x
	# func planta-dos motores do robo:
	'''
	#cv2.imshow('frame',frame_new)
	cv2.imshow('frame',frame)	
	if pressed_key == ord("z"):
		break
cv2.destroyAllWindows()
capture.release() 
