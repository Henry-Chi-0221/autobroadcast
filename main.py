import cv2
import numpy as np
#from sklearn_extra.cluster import KMedoids

from video_stitcher import VideoStitcher
from camera_motion_control.eptz_control import eptz
from yolov5_detect import yolov5_detect

from util import *
from math import sqrt
from time import sleep
import sys
import argparse

import threading
import queue
import os 
from glob import glob
from calibration import calibrate

from motor_driver.arduino_control import ArduinoControl
from motor_driver.coordinate2angle import Coordinate2Angle

class targeting(object):
    def __init__(self , 
                width ,
                height, 
            ):
        self.width = width
        self.height = height

        self.x = width // 2
        self.y = height // 2
        
        ############ 
        # Parameters !!!
        self.center_thres = 800
        self.ball_player_dist_thres = 500
        self.ball_missing_timeout = 120

        ball_sample_rate = 60
        ball_energy_thresh = 40
        ############
        #self.ball_motion_detection = ball_motion_detection( n = ball_sample_rate , thres = ball_energy_thresh )

        self.is_ball_exists = False
        
        self.ball_missing_count = 0

        self.ball_pos = [self.width//2,self.height//2]

        self.x_boundary = [0,self.width]
        self.y_boundary = [0,self.height]

        self.exclude_list = np.array([[434,332],[498,331]])
        self.exclude_list = self.exclude_list*4
        #print(self.exclude_list)

        self.ball_stabilizer = stabilizer( n = 10)
        self.player_stabilizer = stabilizer( n = 20)
        self.target_stabilizer = stabilizer( n = 10)
    def run(self,res ,img):
        self.img = img
        players = []
        
        self.is_ball_exists = False
        if len(res)>0:
            for i ,c in enumerate(res):
                if c[5] == 3.0:
                    players.append(((c[0] + c[2])//2,(c[1] + c[3])//2))
                
                if c[5] == 1.0 and \
                    self.y_boundary[0] < (c[1] + c[3])//2 < self.y_boundary[1] and \
                    self.x_boundary[0] < (c[0] + c[2])//2 < self.x_boundary[1] and \
                    c[4] > 0.6:
                    
                    if self.exclude(((c[0] + c[2])//2,(c[1] + c[3])//2) , self.exclude_list , 100):
                        self.is_ball_exists = True
                        self.ball_missing_count = 0

                        ball_pos = [(c[0] + c[2])//2,(c[1] + c[3])//2]
                        self.ball_pos = ball_pos
                        #self.ball_motion_detection.update(ball_pos)

            if not self.is_ball_exists:
                self.ball_missing_count +=1
            

            if len(players) == 0:
                players = [(self.width//2,self.height//2)]

            player_center = np.mean(np.array(players),axis=0).astype(int)
            ball_stabilized = self.ball_stabilizer.run(np.array(self.ball_pos).astype(int))
            player_stabilized = self.player_stabilizer.run(player_center)
            target = np.mean(np.stack((ball_stabilized , player_stabilized)),axis=0).astype(int)
            target_stabilized = self.target_stabilizer.run(target)
            self.x, self.y = target_stabilized
            """
            cv2.circle(self.img ,player_center , 15,(255,0,0),2)
            cv2.circle(self.img ,player_stabilized , 15,(255,0,0),-1)

            cv2.circle(self.img ,np.array(self.ball_pos).astype(int) , 15,(0,255,255),2)
            cv2.circle(self.img ,ball_stabilized , 15,(0,255,255),-1)

            cv2.circle(self.img ,target , 15,(0,255,0),2)
            cv2.circle(self.img ,(self.x, self.y) , 15,(0,255,0),-1)
            """
            
        cv2.line(img , (0,self.y_boundary[0]),(self.width,self.y_boundary[0]) , (255,0,0) ,5)
        cv2.line(img , (0,self.y_boundary[1]),(self.width,self.y_boundary[1]) , (255,0,0) ,5)
        
        cv2.line(img , (self.x_boundary[0],0),(self.x_boundary[0] , self.height) , (255,0,0) ,5)
        cv2.line(img , (self.x_boundary[1],0),(self.x_boundary[1] , self.height) , (255,0,0) ,5)

        return self.x,self.y

    def draw(self,res,stitched_img,color = (255,255,255)):
        if len(res)>0:
            for i ,c in enumerate(res):
                if c[5] == 1.0:
                    color = (255,0,0)
                else:
                    color = (255,255,255)
                plot_one_box(c[:4] , stitched_img , label= f"{c[5]} : {round(c[4]*100,1)}"  , color= color, line_thickness=3)
    
    def check_distance(self,pt1 ,pt2):
        x1,y1 = pt1[0],pt1[1]
        x2,y2 = pt2[0],pt2[1]
        return sqrt(((x2-x1)**2) + ((y2-y1)**2))
        
    def most_frequent(self,List):
        return max(set(List), key = List.count)
        
    def exclude(self,pt1 , exclude_list ,thres):
        for i in exclude_list:
            k = self.check_distance(pt1 , i)
            if k < thres:
                return False
        return True
        
class autobroadcast(object):
    def __init__(self , 
                model_path , 
                left_path  ,
                right_path , 
                camera_size=[1920,1080] ,
                eptz_boundary = [(0,1920),(0,1080)],
                display = False ,
                single_cam = False,
                record = False,
                sensitivity = 1.0,
                index = None,
                calib = False,
                use_motor  = False
        ):
        self.index = index
        self.cap_L = cv2.VideoCapture(left_path)
        self.cap_R = cv2.VideoCapture(right_path)
        self.single_cam = single_cam
        if not single_cam:
            full_size = int(self.cap_L.get(3)*2) , int(self.cap_L.get(4))
            self.stitcher = VideoStitcher(fullsize=full_size , initial_frame_count=2)
        else:
            full_size = int(self.cap_L.get(3)) , int(self.cap_L.get(4))
        #full_size = (1920,1080)
        self.full_size = full_size

        self.length = int(self.cap_L.get(cv2.CAP_PROP_FRAME_COUNT))

        self.tg = targeting(width = full_size[0] , height=full_size[1])
        
        if cv2.imwrite("black.png" , np.zeros([full_size[1],full_size[0],3])):
            self.detector = yolov5_detect(
                                source="black.png",
                                detect_mode='frame_by_frame',
                                nosave=True,
                                fbf_output_name="output",
                                weights=model_path,
                                imgsz=(1280,1280), # (1280,1280) : 45ms , (640,640) : 16 ms
                                half=True,
                                fbf_close_logger_output=True
                            )
            self.detector.conf_thres = 0.2
        
        self.eptz_control = eptz(
                                size = camera_size, 
                                fullsize = full_size,
                                kp = sensitivity * 0.01,
                                ki = sensitivity * 0.05,
                                kd = sensitivity * 0.08,
                                boundary_offset=eptz_boundary,
                                debug = True
                            )
        
        self.target = full_size[0]//2 , full_size[1]//2
        #self.target = 2286//2 , 1080//2
        self.display = display
        self.record = record
        if self.record:
            #self.recorder = recorder((full_size[0]//2 , full_size[1]//2) , (camera_size[0]//2 , camera_size[1]//2) ,self.cap_L.get(5)//2 , index=index)
            self.recorder = recorder((full_size[0]//2 , full_size[1]//2) , (camera_size[0]//2 , camera_size[1]//2) ,self.cap_L.get(5)//2 , stereo=True)
        self.zoom_ratio_init = None
        if calib :
            self.calib = calibrate(img_path="chessboards/config_2/*")

        self.use_motor = use_motor
        if self.use_motor:
            self.Coor2Angle = Coordinate2Angle(left_angle= -30, right_angle = 30)
            self.arduino = ArduinoControl()
            sleep(5)
            
    def run(self):
        count = 0
        while(self.cap_L.isOpened()):
            # skipping frames
            
            if count%5 != 0:
                count +=1
                ret_L , frame_L = self.cap_L.read()
                _     , frame_R = self.cap_R.read()
                continue
            
            t = timer()
            
            ret_L , frame_L = self.cap_L.read()
            _     , frame_R = self.cap_R.read()
            t.add_label("read")
            
            if ret_L :
                count +=1
                
                #stitch
                if not self.single_cam:
                    stitched_img = self.stitcher.stitch([frame_L ,frame_R]) #cpu : 52.0 ms  ,FPS : 19.059 ; gpu : 59.0 ms  ,FPS : 16.876 
                else:
                    stitched_img = frame_L

                #stitched_img = self.calib.run(stitched_img)

                t.add_label("Image stitching")
                
                #Object detection
                self.detector.run(stitched_img) 
                res = self.detector.pred_np
                t.add_label("Object detection")

                # Logic part
                self.target = self.tg.run(res ,stitched_img)
                
                self.tg.draw(res,stitched_img=stitched_img)
                t.add_label("Targeting")
                
                # ePTZ
                if self.use_motor:
                    t_a = timer()
                    cv2.circle(stitched_img ,self.target , 30,(255,0,0),-1)
                    self.send_to_motor(self.target[0])
                    #t_a.show()
                else:
                    width_for_zoom =  0.01
                    if self.zoom_ratio_init is None:
                        if_not_detect_init = 2.5
                        self.zoom_ratio_init = self.check_zoom_ratio_init(res)
                        zoom_range = (if_not_detect_init,if_not_detect_init*2)
                    else:
                        zoom_range = (self.zoom_ratio_init,self.zoom_ratio_init*2)
                        
                    zoom_value = self.eptz_control.zoom_follow_x(target = (self.eptz_control.current_x ,self.eptz_control.current_y) ,
                                                                zoom_range = zoom_range,
                                                                width_for_zoom = width_for_zoom,
                                                                img = stitched_img
                                                                )
                    src , resized = self.eptz_control.run(stitched_img , zoom_value, self.target[0] , self.target[1])

                t.add_label("ePTZ")
                #prog.update( self.index ,round(count / self.length * 100  , 2))

                if self.record and self.use_motor:
                    #self.recorder.write(stitched_img , resized)
                    self.recorder.write(cv2.resize(stitched_img , (stitched_img.shape[1]//2 , stitched_img.shape[0]//2  )) , cv2.resize(resized , (resized.shape[1]//2 , resized.shape[0]//2  )))
            
            if self.display:
                if not self.use_motor:
                    cv2.imshow('tracked' ,cv2.resize(resized , (resized.shape[1]//2 , resized.shape[0]//2  )) )
                cv2.imshow('stitched' , cv2.resize(stitched_img , (stitched_img.shape[1]//2 , stitched_img.shape[0]//2  )) )
                if cv2.waitKey(1) & 0xff==ord('q'):
                    break
            
            t.add_label("Display")
            #t.show()
            
            if(count / self.length >= 1):
                self.send_to_motor(self.full_size[0]//2)
                sleep(3)
                break
        if self.record:
            self.recorder.release()

    def check_zoom_ratio_init(self,det):
        basket_position = []
        for i in det:
            if i[5] == 2.0:
                basket_position.append([(i[0] + i[2])//2,(i[1] + i[3])//2])
        basket_position = np.array(basket_position)
        if len(basket_position) == 2 :
            basket_distance = np.sqrt(np.sum(np.square(np.absolute(basket_position[0] - basket_position[1]))))   
            return  self.full_size[0] / ( basket_distance * 1.0 ) 
        return None
    
    def send_to_motor(self,point):
        point = remap(point,0,self.full_size[0] , -100,100)
        deg = self.Coor2Angle(point)
        self.arduino.write(deg)
        from_arduino = self.arduino.read()
        del from_arduino
        #print(self.arduino.read().decode("utf-8"))

def test_run(f , idx):
    single_cam = True
    #left_path = f'./videos/vid_{i}/out_L.mp4'

    #left_path = f"/home/bucketanalytics/Desktop/0823/GX0100{i}.mp4"
    left_path = f
    
    if single_cam :
        right_path = f'./videos/vid_{i}/out_L.mp4'
    else:
        right_path = f'./videos/vid_{i}/out_R.mp4'

    model_path = './models/0725_best_model.engine'
    width ,height = 3840 ,2160
    

    bs = autobroadcast(model_path = model_path,
                        left_path = left_path,
                        right_path = right_path,
                        camera_size = [width,height],
                        eptz_boundary = [(0,width),(0,height)],
                        display = True,
                        single_cam=single_cam,
                        record = False,
                        sensitivity=10,
                        index = idx,
                        use_motor = False)

    bs.run()

class Worker(threading.Thread):
    def __init__(self, queue,path):
        threading.Thread.__init__(self)
        self.queue = queue
        self.videos = sorted(glob(path))
    def run(self):
        while self.queue.qsize() > 0:
            idx = self.queue.get() 
            test_run(self.videos[idx],idx)
if __name__ == "__main__":
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_cam", help="Use single image instead of stitched stereo image",action="store_true")
    parser.add_argument("--input","-i", help="Input videos or stream" ,type=str)
    parser.add_argument("--model_path","-i", help="Input videos or stream" ,type=str)

    args = parser.parse_args()

    print(args.single_cam)
    print(args.input)
    """
    t = timer()
    path = "/home/bucketanalytics/Desktop/0826/*MP4"
    #path = "./videos/vid_77/*.mp4"
    num_workers = os.cpu_count()
    num_workers = 1
    #if len(glob(path)) < num_workers:
    #    num_workers = len(glob(path))
    my_queue = queue.Queue()
    prog = progress(num_workers)

    for i in range(len(glob(path))):
        my_queue.put(i)
    workers = []
    for i in range(num_workers):
        workers.append(Worker(my_queue, path))
        workers[i].start()
    for i in workers:
        i.join()
    print("Done.")
    t.show()

"""
titil -> Auto broadcast system
introduction
    - Goal 
structure 
    - Image stitching (optional)
    - Object detection (YOLOv5)
        - Used classes : player , ball , basket
    - Targeting module (strategy)
        - player : Average of center points of all players
        - ball : The center point of ball
        - stabilizer  : Get the average value from the past n frames to present (de-noising)
        - final point : 
            S = stabilizer()
            final = S( S(ball) + S(player) ) 
    - camera motion controller (PID)
        - Demo of PID controller 
Demo 
    - Debug mode
    - Normal mode
    - W/Wo stabilizer
Limitation : 
    - Only one ball is accepted on the field
    - Reliability of yolov5 model
    - Video quality may drop
"""