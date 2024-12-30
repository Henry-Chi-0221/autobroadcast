# Auto Broadcasting System

## System Overview

Auto Broadcasting System uses OpenCV and Deep Learning to track basketball game dynamics. It detects and tracks players, balls, and baskets for real-time analysis. The system supports image stitching, object detection, and targeting with stabilizers for smooth performance. An ePTZ system handles camera motions, making it suitable for live broadcasting and analysis.

# System Output (Demo)

The system generates two outputs:

1. **Resized Output** - Shows the zoomed-in and cropped view based on the tracked target, optimized for viewing specific dynamics in the game.
2. **Source Video** - Displays the original input video stream with annotations like bounding boxes around detected objects.
 

https://github.com/user-attachments/assets/e431d875-abbd-452f-b10c-ddced1bb020c



https://github.com/user-attachments/assets/b88bda37-8866-400b-8034-e4ce1a1ce6b1


## Requirements

```bash
pip install -r requirements.txt
```

# Object Detection (Yolov5)

Utilizes the Yolov5 API for object detection. Export your model to 'TensorRT engine' for improved performance.

## Export TensorRT

```bash
cd yolov5
./export.sh ../models/0725_best_model.pt 
```

## Modify Image Size in Yolov5

```bash
# export.sh
python export.py --weights ../models/$1 --include engine --opset 13 --imgsz 1280 1280 --device 0 --half
```

## Usage Example

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
detector.conf_thres = 0.2 # confidence threshold
detector.run(img) 
res = detector.pred_np 
# format : [x1,y1,x2,y2,conf,class]
```

# Targeting Module

Processes image data to determine a specific target point using stabilizers for smoothing.

## Usage

```python
width, height = 3840, 2160
full_size = (width, height)

tg = targeting(width=full_size[0], height=full_size[1])

target = tg.run(res, img) 
# Output: [x, y]
# res format: [x1, y1, x2, y2, conf, class]

tg.draw(res, stitched_img=img) # Draw bounding boxes for debugging
```

## Stabilizer Parameters

```python
self.ball_stabilizer = stabilizer(n=10)
self.player_stabilizer = stabilizer(n=20)
self.target_stabilizer = stabilizer(n=10)
```

# ePTZ System

Simulates electronic pan-tilt-zoom control with PID-based motion tracking.

## Usage

```python
from eptz_control import eptz 

width, height = 3840, 2160
camera_size = (width, height)
full_size = (width, height)
eptz_boundary = [(0, width), (0, height)]

eptz_control = eptz(
                    size=camera_size, 
                    fullsize=full_size,
                    kp=sensitivity * 0.01,
                    ki=sensitivity * 0.05,
                    kd=sensitivity * 0.08,
                    boundary_offset=eptz_boundary,
                    debug=True
                ) 

src, resized = eptz_control.run(frame, zoom_ratio=z, x_pos=x, y_pos=y) 
```

## PID Control Example

```python
from eptz_control import eptz 
import cv2
from time import time
import random

cap = cv2.VideoCapture('test.mp4')
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

sensitivity = 10
zoom_range = (2.5, 2.5*2)
width_for_zoom = 0.01
camera_size = (width, height)
full_size = (width, height)
eptz_control = eptz(
                    size=camera_size, 
                    fullsize=full_size,
                    kp=sensitivity * 0.01,
                    ki=sensitivity * 0.05,
                    kd=sensitivity * 0.08,
                    boundary_offset=eptz_boundary,
                    debug=True
                )

x, y = width//2, height//2
current_time = time()
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    if (time() - current_time) > 2:
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        current_time = time()

        zoom_value = eptz_control.zoom_follow_x(
            target=(eptz_control.current_x, eptz_control.current_y),
            zoom_range=zoom_range,
            width_for_zoom=width_for_zoom,
            img=frame
        )
    src, resized = eptz_control.run(frame, zoom_ratio=zoom_value, x_pos=x, y_pos=y)
    cv2.imshow("src", frame)
    cv2.imshow("resized", resized)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
```

# Camera Calibration

## Save Parameters

```python
c = calibrate(img_path="chessboards/config_2/*")
print("init done")
c.save_img(vid_path, img_path)
```

## Usage

```python
calib = calibrate(img_path="chessboards/config_2/*") 
img = calib.run(img)
```

# Image Stitching (Optional)

## Usage

```python
width, height = 1920, 1080
full_size = (width*2, height)

stitcher = VideoStitcher(fullsize=full_size, initial_frame_count=2)
stitched_img = stitcher.stitch([frame_L, frame_R])
```




