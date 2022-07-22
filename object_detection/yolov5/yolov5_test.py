from matplotlib.pyplot import axis
import torch
import cv2
from glob import glob
from time import time
from torchvision import transforms as trns
import numpy as np
from PIL import Image
import sys
import hubconf as hub

cap = cv2.VideoCapture(glob("../videos/*")[0])
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = hub._create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True).cuda()

w , h = 1920 , 1080
def cv2_to_tensor(src):
    trans = trns.Compose([
        trns.ToTensor(),
        trns.Resize((640,640))
    ])
    out = Image.fromarray(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
    out = torch.unsqueeze(trans(out),0)
    return out

while(cap.isOpened()):
    ret ,frame = cap.read()
    if not ret:
        break
    else:
        s = time()
        tensor = cv2_to_tensor(frame).cuda()
        pred = model(tensor)
        k = pred.pandas().xyxy[0]
        length = len(k['xmin'])
        for i in range(length):
            xmin = int(k['xmin'][i] * (1920/640) )
            ymin = int(k['ymin'][i] * (1080/640) )
            xmax = int(k['xmax'][i] * (1920/640) )
            ymax = int(k['ymax'][i] * (1080/640) )
            cv2.rectangle(frame , (xmin , ymin) , (xmax , ymax) , (255 ,255, 255) , 1)

        #cv2.imshow('src' , frame)
        print(f"{round(time() - s ,3) *1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} ")
        if cv2.waitKey(1) & 0xff==ord('q'):
            break
        


