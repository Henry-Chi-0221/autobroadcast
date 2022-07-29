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
        self.center_thres = 200
        self.ball_player_dist_thres = 200
        self.ball_motion_detection = ball_motion_detection( n=10 , thres=500 )
        self.KMedoids_res = [0,0]

        self.ball_missing_timeout = 10
        self.is_ball_exists = False
        self.ball_missing_count = 0
    def run(self,res):
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
        
        for i in centers:# [ [x1,y1] , [x2,y2]  ]
            cv2.circle(stitched_img ,i.astype(int) , 15,(0,0,255),-1) 
        cv2.circle(stitched_img ,center.astype(int) , 15,(0,255,255),-1)
        #(center.astype(int))
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

if __name__ == "__main__":
    #model_path = './models/HEAVY_basketball.pt'
    #model_path = './models/HEAVY_basketball.engine'
    #model_path = './models/0719_player_heavy.engine'
    #detector = object_detector(model_path=model_path, width=3840 , height=1080 , imgsz=(1280,320))
    model_path = './models/0725_best_model.engine'
    tg = targeting(width = 3840 , height=1080)
    
    detector = yolov5_detect(source="black.png",
                            detect_mode='frame_by_frame',
                            nosave=True,
                            fbf_output_name="output",
                            weights=model_path,
                            imgsz=(320,1280),
                            half=True,
                            fbf_close_logger_output=True
                           )
    
    i = 53
    left_path = f'./videos/vid_{i}/out_L.mp4'
    right_path = f'./videos/vid_{i}/out_R.mp4'
    cap_L = cv2.VideoCapture(left_path)
    cap_R = cv2.VideoCapture(right_path)
    length = int(cap_L.get(cv2.CAP_PROP_FRAME_COUNT))
    stitcher = VideoStitcher(
                        left_video_in_path=f'./videos/vid_{i}/out_L.mp4',
                        right_video_in_path=f'./videos/vid_{i}/out_R.mp4',
                        video_out_path=f'./videos/vid_{i}/out_res.mp4',
                        gpu=False
                        )
    
    eptz_control = eptz(width=1920 , height=1080)

    count = 0
    x,y = 1920,540
    while(cap_L.isOpened() and cap_R.isOpened()):
        # skipping frames
        count +=1
        if count%4 != 0:
            ret_L , frame_L = cap_L.read()
            _     , frame_R = cap_R.read()
            continue
        
        ret_L , frame_L = cap_L.read()
        _     , frame_R = cap_R.read()
        
        if not ret_L:
            break
        else:
            
            stitched_img = stitcher.stitch([frame_L ,frame_R]) #cpu : 52.0 ms  ,FPS : 19.059 ; gpu : 59.0 ms  ,FPS : 16.876 
            
            t = timer()
            detector.run(stitched_img) 
            #t.show()
            res = detector.pred_np

            # Logic part
            x,y= tg.run(res)

            tg.draw(res,stitched_img=stitched_img)
            src , resized = eptz_control.run(stitched_img , 1.1, x , y)
            #t.show()
            
            cv2.imshow('tracked' ,cv2.resize(resized , (1920//2 , 1080//2  )) )
            
            cv2.imshow('stitched' , cv2.resize(stitched_img , (3840//2 , 1080//2  )) )
            print(f"{round(count / length *100)} %")
            cv2.waitKey(1)
            
