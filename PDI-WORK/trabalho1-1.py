# Digital Image Processing
# Student: Samuel Amico
# Number: 20180010181
# Exercise 1.2 - biel.png


import numpy as np
import cv2

image = cv2.imread('biel.png')

height, width, ch = image.shape
print("height - y: ",height,"width - x: ",width)

img = image.copy()

# [coord x-(x0:xf), coord y(y0,yf)]

image_A = img[0:height/2,0:width/2] 
image_B = img[0:width/2,height/2:height]
image_C = img[height/2:height,0:width/2]
image_D = img[height/2:height,width/2:width]

# ROI`s
cv2.imshow("roi A", image_A)
k = cv2.waitKey(0)
#cv2.imwrite('roiA.png',image_A)
cv2.imshow("roi B", image_B)
k = cv2.waitKey(0)
#cv2.imwrite('roiB.png',image_B)
cv2.imshow("roi C", image_C)
k = cv2.waitKey(0)
#cv2.imwrite('roiC.png',image_C)
cv2.imshow("roi D", image_D)
k = cv2.waitKey(0)
#cv2.imwrite('roiD.png',image_D)


# 1: B/A & D/C:
image[0:width/2,0:height/2] = image_B
image[0:width/2,height/2:height] = image_A
image[height/2:height,0:width/2] = image_D
image[height/2:height,width/2:width] = image_C

#cv2.imshow("1: B/A & D/C", image)
#k = cv2.waitKey(0)
cv2.destroyAllWindows()

# 2: B/A & C/D:
image[0:width/2,0:height/2] = image_B
image[0:width/2,height/2:height] = image_A
image[height/2:height,0:width/2] = image_C
image[height/2:height,width/2:width] = image_D

#cv2.imshow("2: B/A & C/D", image)
#k = cv2.waitKey(0)
#cv2.destroyAllWindows()

# 3: A/B & D/C:
image[0:width/2,0:height/2] = image_A
image[0:width/2,height/2:height] = image_B
image[height/2:height,0:width/2] = image_D
image[height/2:height,width/2:width] = image_C

#cv2.imshow("3: A/B & D/C", image)
#k = cv2.waitKey(0)
#cv2.destroyAllWindows()

#4: A/B & C/D:
image[0:width/2,0:height/2] = image_A
image[0:width/2,height/2:height] = image_B
image[height/2:height,0:width/2] = image_C
image[height/2:height,width/2:width] = image_D

#cv2.imshow("4: A/B & C/D", image)
#k = cv2.waitKey(0)
#cv2.destroyAllWindows()

#5: C/D & A/B:
image[0:width/2,0:height/2] = image_C
image[0:width/2,height/2:height] = image_D
image[height/2:height,0:width/2] = image_A
image[height/2:height,width/2:width] = image_B

#cv2.imshow("5: A/B & C/D", image)
#k = cv2.waitKey(0)

#cv2.destroyAllWindows()

#6: C/D & B/A:
image[0:width/2,0:height/2] = image_C
image[0:width/2,height/2:height] = image_D
image[height/2:height,0:width/2] = image_B
image[height/2:height,width/2:width] = image_A

#cv2.imshow("6: C/D & A/B", image)
#k = cv2.waitKey(0)
#cv2.destroyAllWindows()

#7: D/C & A/B:
image[0:width/2,0:height/2] = image_D
image[0:width/2,height/2:height] = image_C
image[height/2:height,0:width/2] = image_A
image[height/2:height,width/2:width] = image_B

#cv2.imshow("7: D/C & A/B", image)
#k = cv2.waitKey(0)
#cv2.destroyAllWindows()

# 8: D/C & B/A:
image[0:width/2,0:height/2] = image_D
image[0:width/2,height/2:height] = image_C
image[height/2:height,0:width/2] = image_B
image[height/2:height,width/2:width] = image_A

cv2.imshow("8: D/C & B/A", image)
k = cv2.waitKey(0)
#cv2.imwrite('bielmix.png',image)

cv2.destroyAllWindows()
