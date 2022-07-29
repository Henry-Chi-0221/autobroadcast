import cv2
import numpy as np
from sklearn_extra.cluster import KMedoids

from video_stitching.video_stitcher import VideoStitcher
from camera_motion_control.eptz_control import eptz
from yolov5_detect import yolov5_detect

from util import plot_one_box , timer ,ball_motion_detection
from math import sqrt

class targeting(object):
    def __init__(self , width ,height):
        self.x = width // 2
        self.y = height // 2
        
        self.ball_motion_detection = ball_motion_detection( n=10 , thres=500 )
        self.KMedoids_res = [0,0]

        self.center_thres = 200
        self.ball_player_dist_thres = 200
        self.ball_missing_timeout = 10
        self.is_ball_exists = False
        self.ball_missing_count = 0

    def run(self,res ,img):
        self.img = img
        players = []
        self.is_ball_exists = False
        if len(res)>0:
            for i ,c in enumerate(res):
                if c[5] == 3.0:
                    players.append(((c[0] + c[2])//2,(c[1] + c[3])//2))
                    
                if c[5] == 1.0:

                    self.is_ball_exists = True
                    self.ball_missing_count = 0

                    ball_pos = [(c[0] + c[2])//2,(c[1] + c[3])//2]
                    self.ball_motion_detection.update(ball_pos)
                    if self.ball_motion_detection.check_score():
                        #print(self.KMedoids_res , ball_pos ,round(self.ball_motion_detection.ball_score), "target : ball")
                        #cv2.putText(stitched_img, "target : ball", (80, 80), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 255), 2, cv2.LINE_AA)
                        self.x,self.y = ball_pos
                    else:
                        if len(self.KMedoids_res) > 0 and len(ball_pos)>0:
                            #print(self.KMedoids_res , ball_pos ,round(self.ball_motion_detection.ball_score) ,end=" ")
                            dist = self.check_distance(self.KMedoids_res , ball_pos)
                            if dist > self.ball_player_dist_thres:
                                #print("target : ball")
                                #cv2.putText(stitched_img, "target : ball", (80, 80), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 255), 2, cv2.LINE_AA)
                                self.x,self.y = ball_pos
                                #return ball_pos
                            else:
                                #print("target : players")
                                #cv2.putText(stitched_img, "target : players", (80, 80), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 255), 2, cv2.LINE_AA)
                                self.x,self.y = self.KMedoids_res
                                #return self.KMedoids_res     
                    #return self.x,self.y 
                #self.x = (c[0] + c[2])//2
                #self.y = (c[1] + c[3])//2
            if not self.is_ball_exists:
                self.ball_missing_count +=1
                if self.ball_missing_count > self.ball_missing_timeout:
                    #cv2.putText(stitched_img, "target : players", (80, 80), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 255), 2, cv2.LINE_AA)
                    self.x,self.y = self.KMedoids_res
            
            if len(players)>2:
                self.KMedoids_res = self.player_KMedoids(np.asarray(players))
                
        return self.x,self.y

    def player_KMedoids(self,players):
        kmedoids = KMedoids(n_clusters=2, random_state=0)
        centers = kmedoids.fit(players).cluster_centers_

        pred = self.most_frequent(kmedoids.predict(players).tolist())
        
        centers_dist = self.check_distance(centers[0],centers[1])
        if centers_dist < self.center_thres:
            center = np.mean(centers,axis=0)
        else:
            center = centers[pred]
        
        for i in centers:
            cv2.circle(self.img ,i.astype(int) , 15,(0,0,255),-1) 
        cv2.circle(self.img ,center.astype(int) , 15,(0,255,255),-1)
        return center.astype(int)

    def draw(self,res,stitched_img,color = (255,255,255)):
        if len(res)>0:
            for i ,c in enumerate(res):
                if c[5] == 1.0:
                    color = (255,0,0)
                else:
                    color = (255,255,255)
                plot_one_box(c[:4] , stitched_img , label=str(c[5]) , color= color, line_thickness=3)
    
    def check_distance(self,pt1 ,pt2):
        x1,y1 = pt1[0],pt1[1]
        x2,y2 = pt2[0],pt2[1]
        return sqrt(((x2-x1)**2) + ((y2-y1)**2)) 
    def most_frequent(self,List):
        return max(set(List), key = List.count)

class autobroadcast(object):
    def __init__(self , model_path , left_path ,right_path , camera_size=[1920,1080] , display = False):
        self.cap_L = cv2.VideoCapture(left_path)
        self.cap_R = cv2.VideoCapture(right_path)
        full_size = int(self.cap_L.get(3)*2) , int(self.cap_L.get(4))
        self.length = int(self.cap_L.get(cv2.CAP_PROP_FRAME_COUNT))
        self.stitcher = VideoStitcher(fullsize=full_size)

        self.tg = targeting(width = full_size[0] , height=full_size[1])
        
        if cv2.imwrite("black.png" , np.zeros([full_size[0],full_size[1],3])):
            self.detector = yolov5_detect(source="black.png",
                                detect_mode='frame_by_frame',
                                nosave=True,
                                fbf_output_name="output",
                                weights=model_path,
                                imgsz=(320,1280),
                                half=True,
                                fbf_close_logger_output=True
                            )
        
        self.eptz_control = eptz(camera_size , full_size)

        self.target = full_size[0]//2 , full_size[1]//2
        self.display = display
        
    def run(self):
        count = 0
        while(self.cap_L.isOpened() and self.cap_R.isOpened()):
            # skipping frames
            count +=1
            if count%4 != 0:
                ret_L , frame_L = self.cap_L.read()
                _     , frame_R = self.cap_R.read()
                continue
            
            ret_L , frame_L = self.cap_L.read()
            _     , frame_R = self.cap_R.read()
            
            if not ret_L:
                break
            else:
                t = timer()

                #stitch
                stitched_img = self.stitcher.stitch([frame_L ,frame_R]) #cpu : 52.0 ms  ,FPS : 19.059 ; gpu : 59.0 ms  ,FPS : 16.876 
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
                src , resized = self.eptz_control.run(stitched_img , 1.1, self.target[0] , self.target[1])
                t.add_label("ePTZ")
                
                t.show()
                
                if self.display:
                    cv2.imshow('tracked' ,cv2.resize(resized , (resized.shape[1]//2 , resized.shape[0]//2  )) )
                    cv2.imshow('stitched' , cv2.resize(stitched_img , (stitched_img.shape[1]//2 , stitched_img.shape[0]//2  )) )
                    if cv2.waitKey(1) & 0xff==ord('q'):
                        break
                

if __name__ == "__main__":
    #model_path = './models/HEAVY_basketball.pt'
    #model_path = './models/HEAVY_basketball.engine'
    #model_path = './models/0719_player_heavy.engine'
    #detector = object_detector(model_path=model_path, width=3840 , height=1080 , imgsz=(1280,320))
    
    i = 53
    left_path = f'./videos/vid_{i}/out_L.mp4'
    right_path = f'./videos/vid_{i}/out_R.mp4'

    model_path = './models/0725_best_model.engine'

    bs = autobroadcast(model_path = model_path,
                        left_path = left_path,
                        right_path = right_path,
                        camera_size = [1920,1080],
                        display=True)
    bs.run()
            
