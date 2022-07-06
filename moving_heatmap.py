import cv2
from cv2 import addWeighted
import numpy as np
from simple_pid import PID
from time import time
import random

class heatmap(object):
    def __init__(self):
        #initialize
        self.cap = cv2.VideoCapture("./test.mp4")
        self.width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.area = self.width * self.height
        self.ret, self.frame = self.cap.read()
        self.avg = cv2.blur(self.frame, (4, 4))
        self.avg_float = np.float32(self.avg)
        self.moving_x = 320 # range : 0 - 640 
        self.offset_y = 0

        self.cnt_size_thr = 2500 # minimum threshold

        # PID parameters
        self.kp = 0.005
        self.ki = 0.05
        self.kd = 0.01

        #debug toggle
        self.debug = True
        self.last = time()
        self.target = 320
    def preprocess(self):
        self.blur = cv2.blur(self.frame, (5, 5))
        diff = cv2.absdiff(self.avg, self.blur)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        if self.debug:
            self.frame = addWeighted(self.frame , 0.5 , cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) , 0.5,0)
        return thresh

    def pid(self,target):
        des_point = target
        pid = PID( 
                self.kp, 
                self.ki, 
                self.kd, 
                setpoint=des_point,
                output_limits=[-640, 640]
            )

        pid.sample_time = 0.1
        pid_output = pid(self.moving_x+360, dt=pid.sample_time)
        pid_output = int(pid_output)
    
        if ( (pid_output < 0) and (self.moving_x+pid_output > 0) ) or \
        ( (pid_output > 0) and (self.moving_x+pid_output < 640) ):
            self.moving_x += pid_output   
    
    def contours(self,map):
        cnts, _ = cv2.findContours(map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max,id = 2500,0
        for i,c in enumerate(cnts):
            if cv2.contourArea(c) < self.cnt_size_thr:
                continue
            if cv2.contourArea(c) > max:
                max = cv2.contourArea(c)
                id  = i
        if len(cnts) > 0:
            (x, y, w, h) = cv2.boundingRect(cnts[id])
            if(self.debug):
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(self.frame , (x+(w//2),y+(h//2)) ,5 ,(255,0,0) , -1 )
                cv2.drawContours(self.frame, cnts, -1, (0, 255, 255), 2)
            return (x, y, w, h)
        else:
            return False
    
    def run(self):
        while(self.cap.isOpened()):
            start = time()
            self.ret, self.frame = self.cap.read()
            if self.ret==False :
                break
            else:
                map = self.preprocess()
                cnts = self.contours(map)

                if cnts:
                    (x, y, w, h) = cnts
                    des_point = x+(w//2)
                    self.pid(target=des_point)
                    self.moving_x= int(self.moving_x)
                if 0:
                    if time() - self.last >1:
                        self.target = random.randint(0, 1280)
                        self.last = time()
                        print("move to " , self.target)
                    self.pid(target=self.target)
                    self.moving_x= int(self.moving_x)
                
                
                if self.debug:
                    cv2.rectangle(self.frame ,
                    (self.moving_x,180+self.offset_y),
                    (self.moving_x + 640 ,540+self.offset_y),
                    (255, 255, 255),1)

                cv2.imshow("src" , self.frame)
                #cv2.imshow("frame" , cv2.resize(self.frame[180+self.offset_y : 540+self.offset_y, self.moving_x : self.moving_x+640 ],
                                                #(int(self.width) , int(self.height)) ) )
                cv2.imshow("frame" ,self.frame[180+self.offset_y : 540+self.offset_y, self.moving_x : self.moving_x+640 ])                                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.accumulateWeighted(self.blur, self.avg_float, 0.01)
                self.avg = cv2.convertScaleAbs(self.avg_float)
            
            fps = round (1 / (time() - start)  ,3) 
            print("FPS :",fps)
if __name__ == "__main__":
    hmap = heatmap()
    hmap.run()