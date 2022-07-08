import cv2
import numpy as np
from eptz_control import eptz

def remap(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

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
        self.cnt_size_thr = 5000 # minimum threshold
        self.debug = True #debug toggle
        
        
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
            self.frame = cv2.addWeighted(self.frame , 0.5 , cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) , 0.5,0)
        return thresh

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
        eptz_control = eptz(width=self.width , height=self.height)
        while(self.cap.isOpened()):
            self.ret, self.frame = self.cap.read()
            if self.ret==False :
                break
            else:
                map = self.preprocess()
                cnts = self.contours(map)
                if cnts:
                    (x, y, w, h) = cnts
                x_pos = x+(w//2)
                y_pos = y+(h//2)
                z_ratio = remap(w*h , 5000,50000 , 2,1.5)
                src , resized = eptz_control.run(self.frame , z_ratio , x_pos , y_pos)
                cv2.imshow('src' , src)
                cv2.imshow('frame' , resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.accumulateWeighted(self.blur, self.avg_float, 0.01)
                self.avg = cv2.convertScaleAbs(self.avg_float)
        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    hmap = heatmap()
    hmap.run()