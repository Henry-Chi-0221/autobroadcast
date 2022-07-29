import torch
import cv2
from glob import glob
from time import time
from torchvision import transforms as trns
from PIL import Image
import sys
from yolov5 import hubconf as hub
import numpy as np
class object_detector(object):
    def __init__(self , model_path='yolov5s' ,width=1920 , height=1080 ,imgsz = (640,640) ,conf_thres = 0.7):
        #self.model = hub._create(name=model_path, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True).cuda()
        self.model = hub.custom(path=model_path)
        #print(next(self.model.parameters()).is_cuda)
        self.model.conf = conf_thres
        self.width  = width
        self.height = height
        self.imgsz = imgsz
        
    
    def detect(self , tensor):
        s = time()
        if isinstance(tensor , torch.Tensor):
            tensor = self.transform(tensor).cuda()
        elif isinstance(tensor , np.ndarray):
            tensor = self.cv2_to_tensor(tensor).cuda()
        #print(f"{round(time() - s ,3) *1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} ")
        pred = self.model(tensor,size = self.imgsz)
        
        k = pred.pandas().xyxy[0]
        length = len(k['xmin'])
        arr = []
        
        for i in range(length):
            xmin = int(k['xmin'][i] * (self.width/self.imgsz[0]) )
            ymin = int(k['ymin'][i] * (self.height/self.imgsz[1]) )
            xmax = int(k['xmax'][i] * (self.width/self.imgsz[0]) )
            ymax = int(k['ymax'][i] * (self.height/self.imgsz[1]) )
            name = k['name'][i]
            conf = k['confidence'][i]
            arr.append((xmin,ymin,xmax,ymax ,name ,conf)) 
        #print(f"{round(time() - s ,3) *1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} ")
        #
        return arr
        
    def transform(self,img):
        out = torch.unsqueeze(img.permute((2,0,1)),0)
        out = trns.Resize(self.imgsz)(out)
        return out

    def cv2_to_tensor(self,src):
        s = time()
        src = cv2.resize(src , self.imgsz)
        #cv2.imshow('test' , src)
        #cv2.waitKey(0)
        out = Image.fromarray(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
        out = trns.ToTensor()(out)
        out = torch.unsqueeze(out,0)
        #print(f"{round(time() - s ,3) *1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} ")
        return out
if __name__ == "__main__":
    cap = cv2.VideoCapture(glob("./videos/*")[0])

    detector = object_detector(model_path='yolov5s',debug=True)

    while(cap.isOpened()):
        ret ,frame = cap.read()
        if not ret:
            break
        else:
            s = time()
            detector.detect(frame)
            print(f"{round(time() - s ,3) *1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} ")
            


