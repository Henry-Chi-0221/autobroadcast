import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from util import timer
from multiprocessing import Process
#from object_detection.object_detector import object_detector
import numpy as np
import cv2

def main():
    while(1):
        s = timer()
        cv2.warpPerspective(img, M, shape,dst=rot,flags=cv2.INTER_NEAREST)
        #s.show()
        cv2.imshow("test" , rot)
        if cv2.waitKey(1) &0xff==ord('q'):
            break

if __name__ == "__main__":
    model_path = './models/HEAVY_basketball.engine'
    #from yolov5_detect import yolov5_detect
    """
    from yolov5_detect import yolov5_detect
    detector = yolov5_detect(source="./object_detection/videos/20200101_001727_000014.mp4",
                           detect_mode='frame_by_frame',
                           nosave=True,
                           fbf_output_name="output",
                           weights=model_path,
                           imgsz=(320,1280)
                           )
    """
    #detector = object_detector(model_path=model_path, width=3840 , height=1080 , imgsz=(1280,320) ,conf_thres=0.2)
    
    img = np.random.randint(255, size=(4096, 4096, 3))
    img = np.uint8(img)
    rot = np.zeros_like(img)
    M = np.random.randn(3,3)
    shape =(4096, 4096)
    
    main()