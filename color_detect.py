#### Color detect
import numpy as np
import cv2
import time
#from Tkinter import *
#from thread import *
#import time
#import threading



capture = cv2.VideoCapture(0)

def rescale_frame(capturing, wpercent=50, hpercent=50):
    width = int(capturing.shape[1] * wpercent / 100)
    height = int(capturing.shape[0] * hpercent / 100)
    return cv2.resize(capturing, (width, height), interpolation=cv2.INTER_AREA)

def roi_seg(img,hsv):
	low_limit = np.array([80,100,100])     # color (100,50,50)
	upper_limit = np.array([200,255,255]) # color (120,255,255)
	# filtro anti-ruido	
	mask2 = cv2.inRange(hsv,low_limit,upper_limit)
	res = cv2.bitwise_and(img,img,mask=mask2)
	cv2.imshow('res',res)	
	kernel = np.ones((20,20),np.uint8)                  # destruindo os ruidos
	res1 = cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel)
	return res1
	

def filtragem(frame):
	blurred = cv2.GaussianBlur(frame,(11,11),0)
	errosion = cv2.erode(blurred,(11,11),1)
	#cv2.imshow('filter',errosion)
	hsv = cv2.cvtColor(errosion,cv2.COLOR_BGR2HSV)
	roi = roi_seg(frame,hsv)
	return roi
	
def contorno(white_img,frame):
	canny = cv2.Canny(white_img, 50, 200)
	#cv2.imshow('canny',canny)
	# depois tente aplicar contorno no canny
	#ret1,thr = cv2.threshold(white_img, 127, 255, cv2.THRESH_BINARY)
	result = cv2.findContours(canny,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	cont,hierarchy = result if len(result) == 2 else result[1:3]
	if len(cont) > 0:
		areas = [cv2.contourArea(c) for c in cont]
		max_index = np.argmax(areas)
		cont_max = cont[max_index]
		M = cv2.moments(cont[max_index])
		if (M['m00'] != 0):	
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			cv2.circle(frame,(cx,cy),8,(0,255,105),3)
			return (cx,cy)
	return (0,0)


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
erro_old = 0
cont = 0
Y = 100

#def threaded_server():
#	while True:	
#		root = Tk()
#		var = DoubleVar()
#		scale = Scale(root,variable=var)
#		scale.pack(anchor=CENTER)
#		label.Label(root)
#		label.pack()
#		root.mainloop()

	
while True:
	#start_new_thread(threaded_server,())
	_,img = capture.read()
	pressed_key = cv2.waitKey(1) & 0xFF
	frame = rescale_frame(img)
	height,width = frame.shape[:2]
	cx = width/2
	cy = height/2
	cv2.circle(frame,(cx,cy),8,(0,0,255),3)
	roi = filtragem(frame)
	### draw contorno e pegar o centroide:
	(x1,y1)=contorno(roi,frame)
	### controlador:
	kp = 2
	ki = 0.01
	PO = x1
	erro = PO - cx
	cont = cont + erro
	#print('erro=',erro)
	Y = kp*erro+130 + ki*cont*0.1
	cv2.circle(frame,(int(Y),cy),8,(255,10,0),3)
	#####


	#cv2.imshow('frame',frame)	
	if pressed_key == ord("z"):
		break
cv2.destroyAllWindows()
capture.release() 
