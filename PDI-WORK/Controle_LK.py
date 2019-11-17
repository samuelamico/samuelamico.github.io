import numpy as np
import cv2
import time


b = np.array([[]],dtype=np.float32)

# parametros Lucas Ked
lk_params = dict(winSize = (15,15),maxLevel=4,
             criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
# parametros shi - tasu, nao estou utilizando no momentos
feature_params = dict(maxCorners= 100, qualityLevel = 0.3,
                  minDistance = 7, blockSize = 7)

#capture o primeiro frame
capture = cv2.VideoCapture(0)
_,old_img = capture.read()
# convertar em escala cinza e adicione o parametro shi-tasu
old_gray = cv2.cvtColor(old_img,cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
mask = np.zeros_like(old_img)

# pre-pontos para teste e para saber se houve agitacao no gradiente
val = np.array([[45.7,89.6],[45.7,79.6],[35.7,79.6],[35.7,89.6]],dtype=np.float32)

# 5 primeiros sao esquerda, 5 depois direita:
p1 = np.array([[[150.0,140.0],[170.0,140.0],[160,160.0],[150.0,180.0],[170.0,180.0],[450.0,140.0],[470.0,140.0],[460,160.0],[450.0,180.0],
[470.0,180.0],[280.0,300.0],[300.0,300.0],[290.0,320.0],[280.0,340.0],[300.0,340.0],
                [280.0,40.0],[300.0,40.0],[290.0,50.0],[280.0,70.0],[300.0,70.0]]],dtype=np.float32)
inter = val

x_1 = []
x_2 = []
y_1 = []
y_2 = []
# Direita e Esquerda
for i in range(5):
    x_1.append(p1[0,i,0])
for i in range(5,10):
    x_2.append(p1[0,i,0])
for i in range(10,15):
    y_1.append(p1[0,i,1])

##### FUNC PARA CONTROLE
def control(x1,x2,y1,y2,x_1,x_2,y_1,y_2):
    tamanho_x1 = sum(x1)
    tamanho_x_1 = sum(x_1) - 20.0
    if(tamanho_x1 < tamanho_x_1):
        print("direita")
    tamanho_x2 = sum(x2)
    tamanho_x_2 = sum(x_2) + 20.0
    if(tamanho_x2 > tamanho_x_2):
        print("esquerda")
    tamanho_y1 = sum(y1)
    tamanho_y_1 = sum(y_1) + 20.0
    if(tamanho_y1 > tamanho_y_1):
        print("Cima")
    tamanho_y2 = sum(y2)
    tamanho_y_2 = sum(y_2) - 20.0
    if(tamanho_y2 < tamanho_y_2):
        print("Baixo")        
    
    

while capture.isOpened():
        cv2.namedWindow('Controlador')
        ret,img = capture.read()
        gray_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #flag para indicar agitacao no campo que ja pre-estabelecemos do grad
        p_flag = p1
        x1,y1 = (p_flag[0,0,0],p_flag[0,0,1])
        x2,y2 = (p_flag[0,1,0],p_flag[0,1,1])
        x3,y3 = (p_flag[0,2,0],p_flag[0,2,1])
        x4,y4 = (p_flag[0,3,0],p_flag[0,3,1])
        x5,y5 = (p_flag[0,4,0],p_flag[0,4,1])
        new_point,status,error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame,p1,None,**lk_params)
        old_gray = gray_frame.copy()
	#p1 = new_point
	# atualize sua area de interesse:
        t_x1 = []
        t_x2 = []
        b_y1 = []
        b_y2 = []
        for i in range(5):
            t_x1.append(new_point[0,i,0])
        for i in range(5,10):
            t_x2.append(new_point[0,i,0])
        for i in range(10,15):
            b_y1.append(new_point[0,i,1])
        for i in range(15,20):
            b_y2.append(new_point[0,i,1])
            
        # Pinte a regiao do controlador
        for i in range(20):
            #cv2.circle(img,(p1[0,i,0],p1[0,i,1]),5,(0,255,0),-1)
            cv2.circle(img,(new_point[0,i,0],new_point[0,i,1]),5,(255,0,0),-1)
        control(t_x1,t_x2,b_y1,b_y2,x_1,x_2,y_1,y_2)
	
        cv2.imshow('img',img)
        
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord("z"):
                break
cv2.destroyAllWindows()
capture.release() 	
