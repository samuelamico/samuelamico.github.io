from visual import *
import visual as vs   # for 3D panel 
import wx   # for widgets
import cv2
import numpy as np

#### DECECAO DE FACE CONDICOES INICAIS
arqCasc = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(arqCasc)
 
cap = cv2.VideoCapture(1)  #instancia o uso da webcam
centro_x = 0
centro_y = 0
x_real = 0
y_real = 0
x=0
y=0
w=0
h=0
kernel = np.ones((5,5), dtype = "uint8")
posica_x = 0
posica_z = 0

contador_time = 0


##### USO DO lk PARA CONTROLE
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
        return(1)
    tamanho_x2 = sum(x2)
    tamanho_x_2 = sum(x_2) + 20.0
    if(tamanho_x2 > tamanho_x_2):
        print("esquerda")
        return(2)
    tamanho_y1 = sum(y1)
    tamanho_y_1 = sum(y_1) + 20.0
    if(tamanho_y1 > tamanho_y_1):
        print("Cima")
        return(3)
    tamanho_y2 = sum(y2)
    tamanho_y_2 = sum(y_2) - 20.0
    if(tamanho_y2 < tamanho_y_2):
        print("Baixo")
        return(4)
    
    

##### Tamanho do grid
tamanho_x = 100
tamanho_y = 300
tamanho_z = 50

ball_1 = sphere (color = color.white, radius = 0.4)
ball_2 = sphere (color = color.white, radius = 0.4)
ball_3 = sphere (color = color.white, radius = 0.4)
ball_4 = sphere (color = color.white, radius = 0.4)

ball_1.pos = (-38,0,-40)
ball_2.pos = (38,0,40)
ball_3.pos = (38,0,-40)
ball_4.pos = (-38,0,40)

def axes( frame, colour, sz, posn ): # Make axes visible (of world or frame).
                                     # Use None for world.   
    directions = [vs.vector(sz,0,0), vs.vector(0,sz,0), vs.vector(0,0,sz)]
    texts = ["X","Y","Z"]
    posn = vs.vector(posn)
    for i in range (3): # EACH DIRECTION
       vs.curve( frame = frame, color = colour, pos= [ posn, posn+directions[i]])
       vs.label( frame = frame,color = colour,  text = texts[i], pos = posn+ directions[i],
                                                                    opacity = 0, box = False )

axes( None, color.white, 3, (-11,6,0))

# Paredes de cima:
curve(pos=[(-tamanho_x,0,tamanho_z),(tamanho_x,0,tamanho_z)], radius=1.0, color = color.white)
curve(pos=[(-tamanho_x,0,-tamanho_z),(tamanho_x,0,-tamanho_z)], radius=1.0, color = color.white)

curve(pos=[(-tamanho_x,0,tamanho_z),(-tamanho_x,0,-tamanho_z)], radius=1.0, color = color.white)
curve(pos=[(tamanho_x,0,tamanho_z),(tamanho_x,0,-tamanho_z)], radius=1.0, color = color.white)

# Aprendendo a fazer Grid:

curve(pos=[(-tamanho_x,0,-40),(-tamanho_x,0,40)], radius=0.4, color = color.white)


ini_x = -tamanho_x
ini_z = -40
ini_y = 80
step = 13
# Y = 0
# Variando ao longo de Z:
# range( [start], stop[, step] )
'''
for i in range(ini_z,-ini_z,step):
    curve(pos=[(-38,0,i),(38,0,i)], radius=0.4, color = color.white)
# Variando ao longo de X:
for i in range(ini_x,-ini_x,step):
    curve(pos=[(i,0,-40),(i,0,40)], radius=0.4, color = color.white)
'''

# Y Esquerda
for i in range(ini_z,-ini_z,step):
    curve(pos=[(-tamanho_x,tamanho_y,i),(-tamanho_x,0,i)], radius=0.4, color = color.white)
for i in range(0,tamanho_y,step):
    curve(pos=[(-tamanho_x,i,-tamanho_z),(-tamanho_x,i,tamanho_z)], radius=0.4, color = color.white)

# Y DIreita
for i in range(ini_z,-ini_z,step):
    curve(pos=[(tamanho_x,tamanho_y,i),(tamanho_x,0,i)], radius=0.4, color = color.white)
for i in range(0,tamanho_y,step):
    curve(pos=[(tamanho_x,i,-tamanho_z),(tamanho_x,i,tamanho_z)], radius=0.4, color = color.white)

# Y Cima
for i in range(ini_x,-ini_x,step):
    curve(pos=[(i,tamanho_y,-tamanho_z),(i,0,-tamanho_z)], radius=0.4, color = color.white)
for i in range(0,tamanho_y,step):
    curve(pos=[(-tamanho_x,i,-tamanho_z),(tamanho_x,i,-tamanho_z)], radius=0.4, color = color.white)
# Y Baixo
for i in range(ini_x,-ini_x,step):
    curve(pos=[(i,tamanho_y,tamanho_z),(i,0,tamanho_z)], radius=0.4, color = color.white)
for i in range(0,tamanho_y,step):
    curve(pos=[(-tamanho_x,i,tamanho_z),(tamanho_x,i,tamanho_z)], radius=0.4, color = color.white)



## FISICA DO JOGO:

ball = sphere (color = color.green, radius = 2.5, make_trail=True, retain=200)
ball.pos = (3,10,3)
ball.trail_object.radius = 0.05
ball.mass = 2.2
ball.p = vector (-0.15, +9.43, +0.27)


wall = box (pos=(0, 250 , 0), size=(23, 5, 23),  color = color.blue)

dt = 1.5
t=0.0
camera = vector(0,0,0)
scene.forward =  vector(0,-1,0)
contador_x = 0
contador_y =0
num =0
while True:
    rate(100)

    t = t + dt
    ball.pos = ball.pos + (ball.p/ball.mass)*dt

### AQUI SE TEM A DETECCAO DE FACE E SEU MOVIMENTO DE CAMERA
    _, img = cap.read() #pega efeticamente a imagem da webcam
    img = cv2.flip(img,180) #espelha a imagem
    image = cv2.GaussianBlur(img,(5,5),0)
    erosion = cv2.erode(image, kernel, iterations = 1)
    dilation = cv2.dilate(image, kernel, iterations = 1)
    img = dilation
    
    faces = faceCascade.detectMultiScale(
        img,
        minNeighbors=5,
        minSize=(30, 30),
	maxSize=(200,200)
    )
 
    # Desenha um retângulo nas faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    contador_time = contador_time + 1
    if(contador_time > 5):
        centro_x = x + w/2
        centro_y = y + h/2
        contador_time = 0
    ### Posicoes com Transformadas:
    
    cv2.circle(img,(centro_x,centro_y),5,(255,0,0),-1)

    ### Mandar dados da posicao na escala do jogo:
    if ( 89 < centro_x < 316.5):
        posica_x = centro_x - 316.5
    elif(316.7 < centro_x < 543):
        posica_x = centro_x - 316.5

    if(82 < centro_y < 235.5):
        posica_z = centro_y - 235.5
    elif(235.6 < centro_y < 386):
        posica_z = centro_y - 235.5

        
    #cv2.imshow('Video', img) #mostra a imagem captura na janela
 
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

    ## faz a camrea Girar:
    #newforward = rotate(scene.forward, axis=scene.up, angle=math.pi/1000)
    #scene.forward = newforward
    ## Faz a camera Mover:
    '''
    if(t < 10):
        contador_x = contador_x + 0.01
        #contador_y = contador_y + 0.01
    else:
        contador_x = contador_x - 0.01
        #contador_y = contador_y - 0.01        
    scene.center = (contador_x,contador_y,0)
    '''
    contador_x = posica_x
    contador_z = posica_z
    scene.center = (contador_x,0,contador_z)


### AKI SE TEM O CONTROLE:
    ret,img1 = capture.read()
    gray_frame = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
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
    num = control(t_x1,t_x2,b_y1,b_y2,x_1,x_2,y_1,y_2)
    cv2.imshow('contr',img1)
    if(num == 1):
        wall.pos.x = wall.pos.x + 1
    elif(num == 2):
        wall.pos.x = wall.pos.x - 1
    elif(num == 3):
        wall.pos.z = wall.pos.z + 1
    elif(num == 4):
        wall.pos.z = wall.pos.z - 1

### AQUI SE TEM A MECANICA DA BOLA:    
    
    #print("ball=",ball.pos)
    if(ball.y > 300 or ball.y < -7):
        ball.p.y = -ball.p.y
    if(ball.z > tamanho_z or ball.z < -tamanho_z):
        ball.p.z = -ball.p.z
    if(ball.x > tamanho_x or ball.x < -tamanho_x):
        ball.p.x = -ball.p.x
    if( ball.y > wall.y and ball.y < 259):
        if (ball.x < wall.x + 10 and ball.x > wall.x - 10):
            ball.p.y = -ball.p.y














