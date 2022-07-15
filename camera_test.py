import cv2
import numpy as np
from time import time 

cap_1 = cv2.VideoCapture(1)
cap_2 = cv2.VideoCapture(2)
while(cap_1.isOpened() and cap_2.isOpened()):
    start = time()
    ret_1 , frame_1 = cap_1.read()
    ret_2 , frame_2 = cap_2.read()
    print(frame_1.shape , type(frame_1))
    print(frame_2.shape , type(frame_2))
    print("\m")
    cv2.imshow('frame_1',frame_1)
    cv2.imshow('frame_1',frame_2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    fps = round (1 / (time() - start)  ,3) 
    #print(fps)
    
#<class 'numpy.ndarray'> (1080, 3840, 3)