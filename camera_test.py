import cv2
import numpy as np
from time import time 
import os 
from glob import glob

for i in range(10):
    cap_1 = cv2.VideoCapture(2)
    cap_2 = cv2.VideoCapture(1)
    cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    length = len(glob("./videos/*"))+1
    print(f"vid_{length} started !" )
    path = f"./videos/vid_{int(length)}"
    os.mkdir(path) 

    out_1 = cv2.VideoWriter(f'{path}/out_R.mp4', fourcc, 10.0, (1920,  1080))
    out_2 = cv2.VideoWriter(f'{path}/out_L.mp4', fourcc, 10.0, (1920,  1080))
    counter = 0
    while(cap_1.isOpened() and cap_2.isOpened()):
        start = time()
        ret_1 , frame_1 = cap_1.read()
        ret_2 , frame_2 = cap_2.read()
        #print(frame_1.shape , type(frame_1))
        #print(frame_2.shape , type(frame_2))
        #print("\m")

        out_1.write(frame_1)
        out_2.write(frame_2)
        counter += 1
        #print(counter//10)
        if counter > 10 * 60 and 0:
            break
        cv2.imshow('frame_1',frame_1)
        cv2.imshow('frame_2',frame_2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #fps = round (1 / (time() - start)  ,3) 
        #print(fps)
    cap_1.release()
    cap_2.release()
    out_1.release()
    out_2.release()
    cv2.destroyAllWindows()
    print(f"vid_{length} done !" )
#<class 'numpy.ndarray'> (1080, 3840, 3)