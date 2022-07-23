import torch
import cv2
from glob import glob
from time import time
from torchvision import transforms as trns
from PIL import Image
import sys
from yolov5 import hubconf as hub

class object_detector(object):
    def __init__(self , model_path='yolov5s' ,width=1920 , height=1080 ,imgsz = (640,640) ,debug=True):
        
        self.model = hub._create(name=model_path, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True).cuda()
        #self.model = hub.custom( path='yolov5s.engine', device='cuda:0')
        self.width  = width
        self.height = height
        self.imgsz = imgsz
        self.debug = debug
    
    def detect(self , tensor):
        #tensor = self.cv2_to_tensor(img).cuda()
        #img = tensor.cpu().numpy()[:,1920:,:]
        img = tensor.cpu().numpy()
        tensor = self.transform(tensor).cuda()
        pred = self.model(tensor)
        k = pred.pandas().xyxy[0]
        length = len(k['xmin'])
        #print(length)
        arr = []
        for i in range(length):
            xmin = int(k['xmin'][i] * (self.width/self.imgsz[0]) )
            ymin = int(k['ymin'][i] * (self.height/self.imgsz[1]) )
            xmax = int(k['xmax'][i] * (self.width/self.imgsz[0]) )
            ymax = int(k['ymax'][i] * (self.height/self.imgsz[1]) )
            #print(xmin ,xmax ,ymin ,ymax)
            arr.append((xmin ,xmax ,ymin ,ymax))
        
            if self.debug:
                cv2.rectangle(img , (xmin , ymin) , (xmax , ymax) , (255 ,255, 255) , 1)
        return arr
        return None
        if self.debug:
            cv2.imshow('result' , cv2.resize(img , (self.width//2 ,self.height//2 )))
            cv2.waitKey(1)
    
    def transform(self,img):
        out = torch.unsqueeze(img.permute((2,0,1)),0)
        #out = out[:,:,:,1920:]
        out = trns.Resize(self.imgsz)(out)
        #print(out[:,:,:,1920:].shape)
        return out
    def cv2_to_tensor(self,src):
        trans = trns.Compose([
            trns.ToTensor(),
            trns.Resize((640,640))
        ])
        out = Image.fromarray(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
        out = torch.unsqueeze(trans(out),0)
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
            


