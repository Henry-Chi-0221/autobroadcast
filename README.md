# Auto Broadcasting System (ePTZ)
Auto Broadcasting System is a system using OpenCV and Deep-Learning technology to implement a tracking system.

##install

```python
pip install simple-pid
```


## PID control of camera motion

```python
from eptz_control import eptz
import cv2

cap = cv2.VideoCapture('test.mp4') # Use your own test video path instead of this
while(cap.isOpened()):
 
 while(cap.isOpened()):
        ret , frame = cap.read()
        if not ret:
            break
        
        if (time() - current_time )>2:
            #y += 10
            #print(y)
            x = random.uniform(0 , 1280)
            y = random.uniform(0 , 720)
            z = random.uniform(1.5, 3.0)
            current_time = time()
          
        src , resized = eptz_control.run(frame,zoom_ratio=z ,x_pos= x,y_pos= y)
        cv2.imshow("src" , frame)
        cv2.imshow("resized" , resized)
        if cv2.waitKey(1) & 0xff==ord('q'):
            break
```
