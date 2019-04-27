import numpy as np
import cv2
import math
import time




def rescale_frame(capturing, wpercent=50, hpercent=50):
    width = int(capturing.shape[1] * wpercent / 100)
    height = int(capturing.shape[0] * hpercent / 100)
    return cv2.resize(capturing, (width, height), interpolation=cv2.INTER_AREA)
    


def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
	return cv2.LUT(image,table)
	
image = cv2.imread('ponte.jpg')


gamma = 2.8
adjusted = adjust_gamma(image,gamma=gamma)
adjusted_gray = cv2.cvtColor(adjusted,cv2.COLOR_BGR2GRAY)
cv2.imshow("gamma image",adjusted_gray)

k = cv2.waitKey(0)

cv2.imwrite("ponte_iluminada.jpg",adjusted_gray)
cv2.destroyAllWindows()
