import cv2
import numpy as np
 
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
contador_time = 0
posica_x = 0
posica_z = 0
kernel = np.ones((5,5), dtype = "uint8")
while True:
    _, img = cap.read() #pega efeticamente a imagem da webcam
    img = cv2.flip(img,180) #espelha a imagem
    image = cv2.GaussianBlur(img,(5,5),0)
    erosion = cv2.erode(image, kernel, iterations = 1)
    dilation = cv2.dilate(image, kernel, iterations = 1)
    img = dilation
    contador_time = contador_time + 1
    faces = faceCascade.detectMultiScale(
        img,
        minNeighbors=5,
        minSize=(30, 30),
	maxSize=(200,200)
    )
 
    # Desenha um retângulo nas faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
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

    print("posicao x verdadeira e adptada: ", centro_x,posica_x)
    print("posicao z verdadeira e adptada: ", centro_y,posica_z)
        
    cv2.imshow('Video', img) #mostra a imagem captura na janela
 
    #o trecho seguinte é apenas para parar o código e fechar a janela
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
 
cap.release() #dispensa o uso da webcam
cv2.destroyAllWindows() #fecha todas a janelas abertas
