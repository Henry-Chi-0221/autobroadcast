import cv2
import numpy as np
from time import time 

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    start = time()
    ret , frame = cap.read()
    print(frame.shape , type(frame))
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    fps = round (1 / (time() - start)  ,3) 
    #print(fps)
    
#<class 'numpy.ndarray'> (1080, 3840, 3)