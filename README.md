# Auto Broadcasting System 

Auto Broadcasting System is a system using OpenCV and Deep-Learning technology to implement a tracking system.

## Requirements

```python
pip install -r requirements.txt
```



# Object detection ( Yolov5 ) 
Use Yolov5 API for object detection, before you run this, please export your model to 'TenserRT engine' first to have a better performance in speed.
## Export tensorrt
```bash
cd yolov5
./export.sh ../models/0725_best_model.pt 
```
## Change Img_size of yolov5
```bash
# export.sh
python export.py --weights ../models/$1 --include engine --opset 13 --imgsz 1280 1280 --device 0 --half
```
## Usage
```python
model_path = "./models/0725_best_model.engine"

if cv2.imwrite("black.png" , np.zeros([full_size[1],full_size[0],3])):
    detector = yolov5_detect(
                        source="black.png",
                        detect_mode='frame_by_frame',
                        nosave=True,
                        fbf_output_name="output",
                        weights=model_path,
                        imgsz=(1280,1280),
                        half=True,
                        fbf_close_logger_output=True
                    )
detector.conf_thres = 0.2 # set confidence threshold

detector.run(img) 
res = detector.pred_np 
# format : [x1,y1,x2,y2,conf,class]
```


# Targeting module
Targeting module aims to solve a unique point in the image, and the point is the target point of the camera.
## Usage
```python
width ,height = 3840 ,2160
full_size = (width , height)

tg = targeting(width = full_size[0] , height=full_size[1])

target = tg.run(res ,img) 
# return one point : [x,y]
# res is the output of yolov5 
# format : [x1,y1,x2,y2,conf,class]

tg.draw(res,stitched_img=img) #draw bounding boxes on the image, for debug
```
## Set parameters for stabilizers
```python
# In the initialization stage of targeting module 
...

self.ball_stabilizer = stabilizer( n = 10)
self.player_stabilizer = stabilizer( n = 20)
self.target_stabilizer = stabilizer( n = 10)

# Larger n for more stable and more delay.
...
``` 
# ePTZ system

## Usage
```python
#import eptz_control module
from eptz_control import eptz 

# create an instance with fixed width and height

width ,height = 3840 ,2160
camera_size = (width , height)
full_size = (width , height)
eptz_boundary = [(0,width),(0,height)]

eptz_control = eptz(
                    size = camera_size, 
                    fullsize = full_size,
                    kp = sensitivity * 0.01,
                    ki = sensitivity * 0.05,
                    kd = sensitivity * 0.08,
                    boundary_offset=eptz_boundary,
                    debug = True
                ) 

# x and y are the coordinates of the tracking target, and z is the zooming ratio
src , resized = eptz_control.run(frame,zoom_ratio=z ,x_pos= x,y_pos= y) 

#src : original image
#resized : cropped image

```

## PID control of camera motion

```python
# An example of 3-axis PID control

from eptz_control import eptz #import eptz_control module

import cv2
from time import time
import random

cap = cv2.VideoCapture('test.mp4')# Use your own test video path instead of this

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

sensitivity = 10
zoom_range = ( 2.5, 2.5*2 )
width_for_zoom = 0.01
camera_size = (width , height)
full_size = (width , height)
eptz_control = eptz(
                    size = camera_size, 
                    fullsize = full_size,
                    kp = sensitivity * 0.01,
                    ki = sensitivity * 0.05,
                    kd = sensitivity * 0.08,
                    boundary_offset=eptz_boundary,
                    debug = True
                )
x, y = width//2, height//2  # inital values 
current_time = time()
while(cap.isOpened()):
    ret , frame = cap.read()
    if not ret:
        break
    if (time() - current_time ) > 2:  # Generating x,y values randomly every 2 seconds
        x = random.uniform(0 , width) 
        y = random.uniform(0 , height)
        current_time = time()
        
        zoom_value = eptz_control.zoom_follow_x(
            target = (
                eptz_control.current_x ,  
                eptz_control.current_y
            ) ,
            zoom_range = zoom_range,
            width_for_zoom = width_for_zoom,
            img = frame
            )
    src , resized = eptz_control.run(frame,zoom_ratio=zoom_value ,x_pos= x,y_pos= y)
    cv2.imshow("src" , frame)
    cv2.imshow("resized" , resized)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
```
# Camera calibration
## Save parameter for camera

```python
# img_path : desired path to load chessboard
c = calibrate(img_path="chessboards/config_2/*")
print("init done")
c.save_img(vid_path , img_path)
# vid_path : desired path to load input video
# img_path : desired path to save chessboard

#c.test()
```
## Usage
```python
# img_path : desired path to load chessboard
calib = calibrate(img_path="chessboards/config_2/*") 
img = calib.run(img)
```

# Image stitching (optional)
## Usage
```python
width , height = 1920,1080
full_size = (width*2 , height)

#Init
stitcher = VideoStitcher(fullsize=full_size , initial_frame_count=2)

# Run stitching
stitched_img = self.stitcher.stitch([frame_L ,frame_R])
```