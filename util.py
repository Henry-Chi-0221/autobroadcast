import random
import cv2
from matplotlib.pyplot import axis
import numpy as np
from time import time,sleep
import sys
class timer(object):
    def __init__(self):
        self.start = time()
    def reset(self):
        self.start = time()
    def show(self):
        print(f"{round(time() - self.start ,3) * 1000} ms  ,FPS : { round(1 / (time() - self.start ) , 3)} ")

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [0, 0, 0],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

class ball_motion_detection(object):
    def __init__(self,n ,thres):
        self.ball_score = 0
        self.ball_thres = thres
        self.n = n # number of elements for sliding window
        self.arr = [] #sliding window
    def update(self,pos):
        if len(self.arr)<self.n:
            self.arr.append(pos)
        else:
            np_arr = np.array(self.arr)
            std = np.std(np_arr ,axis=0)
            std = np.sum(std) 
            self.ball_score = std
            self.arr.pop(0)
            self.arr.append(pos)
    def check_score(self):
        return  self.ball_score>self.ball_thres
            

if __name__ == "__main__":
    current = time()
    bmd = ball_motion_detection(n=5)
    pos = np.random.rand(2).tolist()

    while(1):
        if time() - current > 2:
            print("move !!!")
            pos = np.random.rand(2).tolist()
            print(pos)
            current = time()
        #print(pos)
        #pos = [1,20]
        #print(pos)
        bmd.update(pos)
        print(bmd.ball_score)
        sleep(0.1)