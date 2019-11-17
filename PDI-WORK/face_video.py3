import cv2
import numpy as np 
import dlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imutils import face_utils

video_capture = cv2.VideoCapture(0)
flag = 0

while True:

    ret, frame = video_capture.read()
    resize = cv2.resize(frame, (100, 200), interpolation = cv2.INTER_LINEAR) 
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    rects = face_detect(gray, 1)

    for (i, rect) in enumerate(rects):

        (x, y, w, h) = face_utils.rect_to_bb(rect)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

video_capture.release()
cv2.destroyAllWindows()
