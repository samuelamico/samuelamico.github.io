#### Using Camshift to track a ROI
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
	#roi_hist = cv2.calcHist([hsv_roi], [0],mask,[180],[0,180])
	#depois tente tirar o mask e coloque:
	roi_hist = cv2.calcHist([hsv_roi], [0],None,[16],[0,180])
	return cv2.normalize(roi_hist,roi_hist, 0, 255, cv2.NORM_MINMAX) 
	
	

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
print("Posicione seu objeto a ser detectado - ROI manual")
roi_area = None
#ret,img = capture.read()
#cv2.namedWindow('img')          nao coloque aki pois vai bugars
#cv2.setMouseCallback('img',on_mouse,0)
while capture.isOpened():
	cv2.namedWindow('img')
	cv2.setMouseCallback('img',on_mouse,0)		
	ret,img = capture.read(2)	
	pressed_key = cv2.waitKey(1) & 0xFF 
	if not ret:
		break
	
	if len(boxes)<4:
		orig = img.copy()
		print("entrou no roi")
		while len(boxes) < 4:
			cv2.imshow('img',img)
			cv2.waitKey(0)	
		boxes = np.array(boxes)
		s = boxes.sum(axis = 1)
		t1 = boxes[np.argmin(s)]
		br = boxes[np.argmax(s)]
		roi = orig[t1[1]:br[1],t1[0]:br[0]]
		#roiBox_x = (t1[0],t1[1])
		#roiBox_y = (br[0],br[1])
		#cv2.imshow('roi',roi)
		roi_area = (t1[0],t1[1],br[0],br[1])
		roi_hist = ROI_segment(roi,img)		
		flag = False
	if len(boxes)>4:
		#print("entrou")
		ret,img = capture.read(2)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1) 
		#apply meanshift:
		ret1,roi_area = cv2.CamShift(dst,roi_area,term_crit)
		#draw it on image:
		#a,b,w,h = track_window
		pts = np.int0(cv2.boxPoints(ret1))
		cv2.polylines(img,[pts],True,255,2)
		#cv2.imshow("Final",img)	

	cv2.imshow('img',img)
	if pressed_key == ord("z"):
		break
cv2.destroyAllWindows()
capture.release() 
	
	

