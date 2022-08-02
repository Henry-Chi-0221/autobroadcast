from simple_pid import PID
import cv2
from time import time
import random
import numpy as np
from util import remap
class eptz(object):
    def __init__(self ,size,fullsize ,kp,ki,kd,debug=False):
        self.width , self.height = size
        self.full_width ,self.full_height = fullsize
        #self.width = int(width)
        #self.height = int(height)

        # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.current_x = self.width//2
        self.current_y = self.height//2
        self.current_zoom = 1

        self.debug = debug
    def pid(self,target,current):
        des_point = target
        pid = PID( 
                self.kp, 
                self.ki, 
                self.kd, 
                setpoint=des_point,
            )
        pid.sample_time = 0.1
        pid_output = pid(current, dt=pid.sample_time)
        return pid_output
            
    def zoom(self,zoom_ratio = 1  , x_offset =0 , y_offset = 0):
        width = self.width//zoom_ratio
        height = self.height//zoom_ratio
        x_center = x_offset
        y_center = y_offset
        x1 , x2 = x_center - (width//2) , x_center + (width//2)
        y1 , y2 = y_center - (height//2) , y_center + (height//2)
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        return x1,y1,x2,y2

    def run(self ,img, zoom_ratio = 1  , x_pos =0 , y_pos = 0):
        #self.width * ( 1*2 -0.5*(1/self.current_zoom)) = self.full_width - 
        pid_x = self.pid(x_pos , self.current_x)
        if  self.width*0.5*(1/self.current_zoom) < self.current_x + pid_x <   self.full_width - self.width * (0.5*(1/self.current_zoom))  : #here!!
            self.current_x += pid_x
        
        pid_y = self.pid(y_pos , self.current_y) # target current
        if  self.height*0.5*(1/self.current_zoom) < self.current_y + pid_y < self.full_height - self.height * (0.5*(1/self.current_zoom)):
            self.current_y += pid_y
            
        pid_zoom = self.pid(zoom_ratio , self.current_zoom)
        
        #here !!!
        if  (0 < self.current_zoom + pid_zoom < 3) and \
            (self.current_x - self.width*0.5*(1/self.current_zoom) )>= 0  and \
            (self.current_x + self.width*0.5*(1/self.current_zoom) )<= self.full_width and\
            self.current_y - self.height*0.5*(1/self.current_zoom) >= 0  and \
            self.current_y + self.height*0.5*(1/self.current_zoom) <= self.full_height :

            self.current_zoom += pid_zoom
        
        x1,y1,x2,y2 = self.zoom(zoom_ratio=self.current_zoom,x_offset=self.current_x,y_offset=self.current_y) #x
        res_x = np.clip(np.array([x1,x2]) , 0 , self.full_width)
        res_y = np.clip(np.array([y1,y2]) , 0 , self.full_height)
        (x1,x2), (y1,y2) = res_x[:] , res_y[:]
        if self.debug:
            cv2.rectangle(img , (x1,y1) , (x2,y2) , (0,255,255) , 2)
            cv2.circle(img , (int(x_pos) , int(y_pos)) , 15 ,(0,255,0) , -1)
            cv2.circle(img , ((x1+x2)//2 , (y1+y2)//2) , 15 ,(255,255,255) , 2)
        
        resized = cv2.resize(img[y1:y2,x1:x2] , (self.width , self.height))
        
        return img ,resized
        
    def zoom_follow_x(self ,target, zoom_value , width ,height ,zoom_range,width_for_zoom ,debug = False,img = None):
        
        center = width//2 
        
        boundary = center - width_for_zoom*width ,center + width_for_zoom*width
        
        if debug and img is not None:
            cv2.line(img , (int(boundary[0]),0),(int(boundary[0]),height) , (0,255,0) ,5)
            cv2.line(img , (int(boundary[1]),0),(int(boundary[1]),height) , (0,255,0) ,5)
        if not boundary[0]<target[0]<boundary[1]:
            if target[0] > boundary[1]:
                diff_abs = target[0] - boundary[1]
            else:
                diff_abs = boundary[0] - target[0]
            zoom_value = remap(diff_abs , 0,(0.5-width_for_zoom)*width , zoom_range[0],zoom_range[1])
        return zoom_value
if __name__ == "__main__":
    cap = cv2.VideoCapture('test.mp4')
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    eptz_control = eptz(width=width , height=height)
    x ,y,z = 640 , 360 ,2
    current_time = time()

    while(cap.isOpened()):
        ret , frame = cap.read()
        if not ret:
            break
        
        if (time() - current_time )>2:
            x = random.uniform(0 , 1280)
            y = random.uniform(0 , 720)
            z = random.uniform(1.5, 3.0)
            current_time = time()
        s = time()
        src , resized = eptz_control.run(frame,zoom_ratio=z ,x_pos= x,y_pos= y)
        print(f"{round( (time() - s) * 1000,3)} ms , FPS : {round( 1/(time() - s),3)}")
        cv2.imshow("src" , frame)
        cv2.imshow("resized" , resized)
        if cv2.waitKey(1) & 0xff==ord('q'):
            break