import cv2
from time import time ,sleep
from object_detection.object_detector import object_detector
from util import plot_one_box , timer
from memory_profiler import profile
import gc


i = 53
path = f'./videos/vid_{i}/out_R.mp4'
#model_path = './models/yolov5s.engine'
#model_path = './models/yolov5s.pt'
#model_path = './models/HEAVY_basketball.pt'
model_path = './models/HEAVY_basketball.engine'
cap = cv2.VideoCapture(path)
detector = object_detector(model_path=model_path, width=3840 , height=1080 , imgsz=(1280,320) ,conf_thres=0.2)


count = 0
while(cap.isOpened()):
    ret , frame=  cap.read()
    if not ret:
        break
    else:
        count += 1
        if count%2 != 0:
            ret , frame=  cap.read()
            continue
        frame = cv2.resize(frame , (3840,1080))
        t = timer()
        res = detector.detect(frame)
        t.show()
        #print(f"{round(time() - s ,3) * 1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} ")
        for i in res:
            plot_one_box(i[:4] , frame , label=f"{i[4]} {round(i[5]*100 , 2)}%" , color=(255,255,255) , line_thickness=3)

        cv2.imshow("src" , frame)

    if cv2.waitKey(1) &0xff==ord('q'):
        break