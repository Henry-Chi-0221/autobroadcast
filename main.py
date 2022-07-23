import torch
from object_detection.yolov5_test import object_detector
from video_stitching.new import VideoStitcher
import cv2
from glob import glob
from time import time
import numpy as np
import torchvision.transforms as trns

def show_tensor_to_PIL(src):
    return trns.ToPILImage()(src).convert("RGB")

def tensor_to_cv2(src , invert=False):
        if invert==True:
            return cv2.cvtColor(np.invert(np.asarray(show_tensor_to_PIL(src))),cv2.COLOR_RGB2BGR)
        else :
            return cv2.cvtColor(np.asarray(show_tensor_to_PIL(src)),cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    detector = object_detector(model_path='yolov5s',debug=False , width=1920 , height=1080 , imgsz=(640,640))
    i = 53
    left_path = f'./videos/vid_{i}/out_L.mp4'
    right_path = f'./videos/vid_{i}/out_R.mp4'
    cap_L = cv2.VideoCapture(left_path)
    cap_R = cv2.VideoCapture(right_path)

    stitcher = VideoStitcher(
                        left_video_in_path=f'./videos/vid_{i}/out_L.mp4',
                        right_video_in_path=f'./videos/vid_{i}/out_R.mp4',
                        video_out_path=f'./videos/vid_{i}/out_res.mp4'
                        )
    #stitcher.run(i)
    while(cap_L.isOpened() and cap_R.isOpened()):
        ret_L , frame_L = cap_L.read()
        _     , frame_R = cap_R.read()
        
        if not ret_L:
            break
        else:
            s = time()
            stitched_img = stitcher.stitch([frame_L ,frame_R])
            #print(stitched_img.shape) # torch.Size([1080, 3840, 3])
            arr = stitched_img.cpu().numpy()

            res = detector.detect(stitched_img[:,1920:,:])
            if res:
                for i in res:
                    xmin ,xmax ,ymin ,ymax = i
                    #print(res)
                    cv2.rectangle(arr[:,1920: ,:] , (xmin,ymin) , (xmax,ymax) , (255,255,255) , 2)
            cv2.imshow('stitched' , cv2.resize(arr , (3840//2 , 1080//2  )) )
            cv2.waitKey(1)
            print(f"{round(time() - s ,3) *1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} ")