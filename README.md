# Auto Broadcasting System (ePTZ)
Auto Broadcasting System is a system using OpenCV and Deep-Learning technology to implement a tracking system.

## requirements

```python
pip install opencv-python
pip install simple-pid 
```
## Usage
```python
#import eptz_control module
from eptz_control import eptz 

# create an instance with fixed width and height
eptz_control = eptz(width=width , height=height) 

# x and y are the coordinates of the tracking target, and z is the zooming ratio
eptz_control.run(frame,zoom_ratio=z ,x_pos= x,y_pos= y) 
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
eptz_control = eptz(width=width , height=height) 
x, y, z = width//2, height//2, 2  # inital values 
current_time = time()
while(cap.isOpened()):
    ret , frame = cap.read()
    if not ret:
        break
    if (time() - current_time )>2:  # Generating x,y,z values randomly every 2 seconds
        x = random.uniform(0 , width) 
        y = random.uniform(0 , height)
        z = random.uniform(1.5, 3.0)
        current_time = time()
    src , resized = eptz_control.run(frame,zoom_ratio=z ,x_pos= x,y_pos= y)
    cv2.imshow("src" , frame)
    cv2.imshow("resized" , resized)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
```

## Demos
* Top: normal mode    
* Bottom: debug mode
* left: global camera 
* right: moving camera



https://user-images.githubusercontent.com/48129098/177932851-64f4fbab-936e-42a1-9d28-907d2d938a59.mp4






