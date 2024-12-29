import random
import cv2
from matplotlib.pyplot import axis
import numpy as np
from time import *
import sys
import os



def remap(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

class timer(object):
    def __init__(self):
        self.start = time()
        self.records = []
    def reset(self):
        self.start = time()
        self.records = []
    def show(self):
        if len(self.records)==0:
            print(f"{round(time() - self.start ,3) * 1000} ms  ,FPS : { round(1 / (time() - self.start ) , 3)} ")
        else:
            sum = 0.0
            for i in self.records:
                s = f"{round(i[0] ,3) * 1000} ms  ,FPS : { round( i[1],3)}  <- {i[2]}"
                print(s)
                sum += i[0]
            print(f"{round(sum ,3) * 1000} ms  ,FPS : { round( 1/sum,3)}  <- All\n")

    def add_label(self,text):
        t = time() - self.start
        fps = 1 / (time() - self.start)
        self.records.append((t,fps,text))
        self.start = time()
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
            z = (pos - np.mean(np_arr, axis=0) ) / std
            """
            print("pos",pos)
            print("mean" ,np.mean(np_arr , axis=0))
            print("std",std)
            print("z",z)
            print("res : ",np.sqrt(np.sum(np.square(z))) ,"old : ", np.sum(std) )
            """
            std = np.sum(std) 
            self.ball_score = std
            self.arr.pop(0)
            self.arr.append(pos)
    def check_score(self):
        return  self.ball_score>self.ball_thres
    
class stabilizer():
    def __init__(self, n):
        self.arr = []
        self.n = n
    def run(self,pos):
        if len(self.arr)<self.n:
            self.arr.append(pos)
            return pos
        else:
            self.arr.pop(0)
            self.arr.append(pos)
            return np.mean(np.array(self.arr) , axis=0).astype(int)
    def get(self):
        if len(self.arr)<self.n:
            return pos
        else:
            return np.mean(np.array(self.arr) , axis=0).astype(int)
class eptz_random(object):
    def __init__(self , x=3840,y=1080,z=1.5):
        self.current = time()
        self.x ,self.y , self.z = x,y,z

    def update(self):
        if (time() - self.current )>5:
            x = random.uniform(0 , self.x)
            y = random.uniform(0 , self.y)
            z = random.uniform(1, self.z)
            self.current = time()
            return x,y,z
        return False
def remap(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

class recorder():
    def __init__(self ,full_size , camera_size, fps , index = None , stereo = False):
        self.stereo = stereo
        t =  strftime("%Y_%m_%d_%H_%M_%S")
        if index is None:
            t1 = f"results/{t}_1.mp4"
            if self.stereo:
                t2 = f"results/{t}_2.mp4"
        else:
            t1 = f"results/{t}_{index}_1.mp4"
            if self.stereo:
                t2 = f"results/{t}_{index}_2.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.out_1 = cv2.VideoWriter(t1, fourcc, fps, camera_size)
        if self.stereo:  
            self.out_2 = cv2.VideoWriter(t2, fourcc, fps, full_size)  

    def write(self ,frame_1 , frame_2):
        self.out_1.write(frame_2)
        if self.stereo:
            self.out_2.write(frame_1)
    def release(self):
        self.out_1.release()
        if self.stereo:
            self.out_2.release()
class remove_black_border():
    def __init__(self):
        pass
    def run(self,img,M ,output_shape):
        white = np.ones_like(img,dtype=np.uint8)*255
        white = cv2.warpPerspective(white, M,output_shape , flags=cv2.INTER_NEAREST )
        img = cv2.warpPerspective(img, M,output_shape , flags=cv2.INTER_NEAREST )
        
        (cnts, _) = cv2.findContours(white[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        (x, y, w, h) = cv2.boundingRect(cnts[0])
        #self.roi = (x, y, w, h)
        return img[y:y+h,x:x+w] , img , (x, y, w, h)
class progress():
    def __init__(self,num_workers):
        self.prog = np.zeros([num_workers])
    def update(self,idx,data):
        self.prog[idx] = data
        #os.system("clear")
        for i,p in enumerate(self.prog):
            print(f"Thread_{i} : {p} %")
        
if __name__ == "__main__":
    pass