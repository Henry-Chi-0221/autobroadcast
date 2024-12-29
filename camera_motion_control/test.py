import cv2
import numpy as np
cap1 = cv2.VideoCapture('results/res_2/no_mask/resized.mp4')
cap2 = cv2.VideoCapture('results/res_2/no_mask/src.mp4')
cap3 = cv2.VideoCapture('results/res_2/with_mask/resized.mp4')
cap4 = cv2.VideoCapture('results/res_2/with_mask/src.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out1 = cv2.VideoWriter('./results/res_2/merged.mp4', fourcc, 30.0, (3840,  2160))
while(cap1.isOpened()):
    ret1,frame1 = cap1.read()
    ret2,frame2 = cap2.read()
    ret3,frame3 = cap3.read()
    ret4,frame4 = cap4.read()
    if ret1 and ret2 and ret3 and ret4 :
        no_mask = np.hstack((frame2,frame1))
        with_mask = np.hstack((frame4,frame3))
        merged = np.vstack((no_mask , with_mask))
        #merged = cv2.resize(merged , (1280,720))
        print(merged.shape)
        cv2.imshow('src1' , merged)
        out1.write(merged)
        if cv2.waitKey(1) &0xff == ord('q'):
            break
    else:
        break
out1.release()