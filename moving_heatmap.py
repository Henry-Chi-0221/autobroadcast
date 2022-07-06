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
        self.avg = cv2.blur(self.frame, (5, 5))
        self.avg_float = np.float32(self.avg)
        self.moving_x = 320 # range : 0 - 640 
        self.moving_y = 0

        self.cnt_size_thr = 5000 # minimum threshold

        # PID parameters
        self.kp = 0.01
        self.ki = 0.2
        self.kd = 0.1

        #debug toggle
        self.debug = True
        self.last = time()
        self.target = 320
        
        self.scale = 2
    def zoom_center(self,img , x_offset=0 , y_offset = 0, zoom_factor=2 ):
        y_size = img.shape[0]
        x_size = img.shape[1]
        
        # define new boundaries
        x1 = int(0.5*x_size*(1-1/zoom_factor) + 0.5 * x_offset)
        x2 = int(x_size-0.5*x_size*(1-1/zoom_factor) + 0.5 * x_offset)
        y1 = int(0.5*y_size*(1-1/zoom_factor) + 0.5 * y_offset)
        y2 = int(y_size-0.5*y_size*(1-1/zoom_factor) + 0.5 * y_offset)

        return (x1,y1,x2,y2)
        # first crop image then scale
        #img_cropped = img[y1:y2,x1:x2]
        #return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)

    def preprocess(self):
        self.blur = cv2.blur(self.frame, (5, 5))
        diff = cv2.absdiff(self.avg, self.blur)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.dilate(thresh,kernel,iterations = 3)
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
            )
        current_center = self.moving_x+self.width // (2*self.scale)
        pid.sample_time = 0.1
        pid_output = pid(current_center, dt=pid.sample_time)
        pid_output = int(pid_output)
        print(pid_output ,self.moving_x , pid_output+self.moving_x)

        if  (self.moving_x+(self.width/self.scale)+pid_output ) < self.width and \
            (self.moving_x*self.width*(1-1/self.scale)+pid_output ) > 0 :
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
        if len(cnts) > 0 and max > 2500:
            (x, y, w, h) = cv2.boundingRect(cnts[id])
            if(self.debug):
                
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(self.frame , (x+(w//2),y+(h//2)) ,5 ,(255,0,0) , -1 )
                cv2.drawContours(self.frame, cnts, -1, (0, 255, 255), 2)
            return (x, y, w, h)
        else:
            return False
    
    def run(self):
        (x, y, w, h) = 0,0,0,0
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
                cv2.circle(self.frame , (des_point,360) , 10 , (255,255,255) ,-1)
                self.pid(target=des_point)
                self.moving_x = int(self.moving_x)
                """
                if 0:
                    if time() - self.last >1:
                        self.target = random.randint(0, 1280)
                        self.last = time()
                        print("move to " , self.target)
                    self.pid(target=self.target)
                    self.moving_x= int(self.moving_x)
                """
                x1,y1,x2,y2 = self.zoom_center(self.frame , self.moving_x , self.moving_y , 2 )
                if self.debug or 1:
                    cv2.rectangle(self.frame ,(x1,y1),(x2 ,y2),(255, 255, 255),1)

                cv2.imshow("src" , self.frame)
                #cv2.imshow("frame" , cv2.resize(self.frame[180+self.moving_y : 540+self.moving_y, self.moving_x : self.moving_x+640 ],
                                                #(int(self.width) , int(self.height)) ) )

                cropped = self.frame[y1:y2, x1:x2]
                cv2.circle(cropped , (cropped.shape[1]//2,cropped.shape[0]//2) , 10 , (255,0,255) ,-1)
                cv2.imshow("frame" ,cropped)   

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.accumulateWeighted(self.blur, self.avg_float, 0.01)
                self.avg = cv2.convertScaleAbs(self.avg_float)
            
            #fps = round (1 / (time() - start)  ,3) 
            #print("FPS :",fps)
if __name__ == "__main__":
    hmap = heatmap()
    hmap.run()