from simple_pid import PID
import cv2
from time import time
import random
import numpy as np
from util import remap, timer
from glob import glob
class eptz(object):
    def __init__(self ,size,fullsize ,kp,ki,kd,debug=False ,boundary_offset=[(0,3840),(0,1080)]):
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

        self.boundary_offset = boundary_offset

        self.rotation_border = []
        
        self.debug = debug
    def pid(self,target,current , suppress=False):
        des_point = target
        if suppress:
            self.kp *= 0.1 
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
        
        pid_x = self.pid(x_pos , self.current_x)
        
        x_min = self.width*0.5*(1/self.current_zoom) 
        x_max = self.full_width - self.width * (0.5*(1/self.current_zoom)) 
        x_pred = self.current_x + pid_x

        if  x_min < x_pred <  x_max and \
            self.boundary_offset[0][0] < x_pred < self.boundary_offset[0][1]: # here!!
            self.current_x += pid_x

        if x_pred > x_max :
            self.current_x = x_max

        if x_pred < x_min :
            self.current_x = x_min

        if x_pred + x_min > self.boundary_offset[0][1]:
            self.current_x = self.boundary_offset[0][1] - x_min

        if x_pred - x_min < self.boundary_offset[0][0]:
            self.current_x = self.boundary_offset[0][0] + x_min
        """
        if len(self.rotation_border) > 0 :
            w = (self.rotation_border[0][1] - self.rotation_border[0][0] ) //2
            if self.rotation_border[0][0] <  self.boundary_offset[0][0]:
                self.current_x = self.boundary_offset[0][0] + w

            if self.rotation_border[0][1] >  self.boundary_offset[0][1]:
                self.current_x = self.boundary_offset[0][1] - w
        """
        pid_y = self.pid(y_pos , self.current_y ,suppress=True) # target current
        
        y_min = self.height*0.5*(1/self.current_zoom)
        y_max = self.full_height - self.height * (0.5*(1/self.current_zoom))
        
        y_pred = self.current_y + pid_y

        if  y_min < y_pred < y_max and self.boundary_offset[1][0] < y_pred < self.boundary_offset[1][1]:
            self.current_y += pid_y
        if y_pred > y_max :
            self.current_y = y_max
        if y_pred < y_min :
            self.current_y = y_min
        if y_pred > self.boundary_offset[1][1]:
            self.current_y = self.boundary_offset[1][1]-y_min
        if y_pred < self.boundary_offset[1][0]:
            self.current_y = self.boundary_offset[1][0]-y_min
        pid_zoom = self.pid(zoom_ratio , self.current_zoom)
        
        #here !!!

        if  (0 < self.current_zoom + pid_zoom < 10) and \
            (self.current_x - self.width*0.5*(1/self.current_zoom) ) >= 0  and \
            (self.current_x + self.width*0.5*(1/self.current_zoom) ) <= self.full_width and\
            (self.current_y - self.height*0.5*(1/self.current_zoom) )>= 0  and \
            (self.current_y + self.height*0.5*(1/self.current_zoom) )<= self.full_height :

            self.current_zoom += pid_zoom

        x1,y1,x2,y2 = self.zoom(zoom_ratio=self.current_zoom,
                                x_offset=self.current_x,
                                y_offset=self.current_y) 
                                
        res_x = np.clip(np.array([x1,x2]) , 0 , self.full_width)
        res_y = np.clip(np.array([y1,y2]) , 0 , self.full_height)
        
        (x1,x2), (y1,y2) = res_x[:] , res_y[:]
        
        crop = img[y1:y2,x1:x2]

        #t = timer()
        resized = cv2.resize( crop, (self.width , self.height) ,interpolation=cv2.INTER_NEAREST)
        #t.show()

        """
        f = 3000
        #print(x1 ,x2 ,y1 ,y2)
        xc = (self.boundary_offset[0][1])/2
        yc = (self.boundary_offset[1][1])/2
        _ ,pts = self.rotate_warp(img[y1:y2,x1:x2] , (x1,y1),(x2,y2) ,f ,xc,yc)
        w = (pts[0][2][0] - pts[0][0][0])//2
        self.rotation_border = [[pts[0][0][0] , pts[0][2][0]]]
        """
        
        if self.debug:
            
            cv2.rectangle(img , (x1,y1) , (x2,y2) , (0,255,0) , 2)

            #cv2.circle(img , (int(x_pos) , int(y_pos)) , 15 ,(0,255,0) , -1)
            #cv2.circle(img , ((x1+x2)//2 , (y1+y2)//2) , 15 ,(255,255,255) , 2)
            """
            cv2.line(img , (self.boundary_offset[0][0],0),(self.boundary_offset[0][0],self.full_height) , (255,255,0) ,5)
            cv2.line(img , (self.boundary_offset[0][1],0),(self.boundary_offset[0][1],self.full_height) , (255,255,0) ,5)
            
            cv2.line(img , (0,self.boundary_offset[1][0]),(self.full_width ,self.boundary_offset[1][0]) , (255,255,0) ,5)
            cv2.line(img , (0,self.boundary_offset[1][1]),(self.full_width ,self.boundary_offset[1][1]) , (255,255,0) ,5)
            """
        return img ,resized
        
    def zoom_follow_x(self ,target,zoom_range,width_for_zoom,img = None):
        zoom_value = zoom_range[0]
        #center = width//2 
        width = self.boundary_offset[0][1] - self.boundary_offset[0][0]
        center = ( self.boundary_offset[0][0] + self.boundary_offset[0][1] ) //2
        boundary = center - width_for_zoom*width ,center + width_for_zoom*width
        
        if self.debug and img is not None:
            height = img.shape[1]
            #cv2.line(img , (int(boundary[0]),0),(int(boundary[0]),height) , (0,255,0) ,5)
            #cv2.line(img , (int(boundary[1]),0),(int(boundary[1]),height) , (0,255,0) ,5)
            
        if not boundary[0]<target[0]<boundary[1]:
            if target[0] > boundary[1]:
                diff_abs = target[0] - boundary[1]
            else:
                diff_abs = boundary[0] - target[0]
            zoom_value = remap(diff_abs , 0,(0.5-width_for_zoom)*width , zoom_range[0],zoom_range[1])
        return zoom_value
    
    def transform(self,pt ,f,xc,yc):
        x,y = pt
        x = ( f * np.tan( (x-xc) / f) ) + xc
        y = ( (y-yc) / np.cos( (x-xc) / f ) ) + yc
        x , y = int(x) , int(y)
        return (x,y)
    def rotate_warp(self,img , p1,p2 ,f ,xc,yc):
        white_img = np.ones_like(img ,dtype=np.uint8)*255

        x1,y1 = p1
        x2,y2 = p2

        pt1 = [x1,y1]
        pt2 = [x1,y2]
        pt3 = [x2,y1]
        pt4 = [x2,y2]
        current = np.float32([pt1,pt2,pt3,pt4])
        pt1 = self.transform(pt1, f,xc,yc)
        pt2 = self.transform(pt2, f,xc,yc)
        pt3 = self.transform(pt3, f,xc,yc)
        pt4 = self.transform(pt4, f,xc,yc)
        target = np.float32([pt1,pt2,pt3,pt4])
        
        w ,h = pt4[0] - pt1[0] ,pt4[1] - pt1[1]

        M = cv2.getPerspectiveTransform(current, target)

        processed = cv2.warpPerspective(img,M,(3840, 1080))

        white_img = cv2.warpPerspective(white_img,M,(3840, 1080))
        
        (cnts, _) = cv2.findContours(white_img[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        (x, y, w, h) = cv2.boundingRect(cnts[0])
        cv2.rectangle(white_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imshow( "mask", cv2.resize(white_img[y:y+h,x:x+w],(1920//2,1080//2)))
        processed = cv2.resize(processed[y:y+h,x:x+w] , (1920, 1080))
        
        return processed , [np.array([pt1,pt2,pt3,pt4],np.int32) ]
        

if __name__ == "__main__":


    path = "/home/bucketanalytics/Desktop/0826/*MP4"
    vid = sorted(glob(path))[0]
    cap = cv2.VideoCapture(vid)
    width ,height = 3840 ,2160
    full_size = int(self.cap_L.get(3)) , int(self.cap_L.get(4))
    camera_size = [width,height]
    sensitivity = 8
    self.eptz_control = eptz(
                                size = camera_size, 
                                fullsize = full_size,
                                kp = sensitivity * 0.01,
                                ki = sensitivity * 0.05,
                                kd = sensitivity * 0.08,
                                boundary_offset=eptz_boundary,
                                debug = True
                            )

    x ,y,z = width//2 , height//2 ,1
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
