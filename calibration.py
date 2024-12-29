import cv2
import numpy as np 
import glob
from util import recorder , timer

def show(img,label = "src"):
    h,w = img.shape[:2]
    img = cv2.resize(img , (w//4,h//4))
    cv2.imshow(label ,   img)

class calibrate():
    def __init__(self ,img_path="chessboards/config_1/*"):
        w,h = 3840,2160
        calib_parameter_path = "chessboards/config_2/calibration.npy"
        if len(glob.glob((calib_parameter_path))) == 0 :
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((6*9,3), np.float32)
            objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
            objpoints = [] 
            imgpoints = [] 
            images = sorted(glob.glob(img_path))[:50]
            for i,fname in enumerate(images):
                
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
                print(f"{i+1} / {len(images)}, {fname}, status: {ret}")
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                    imgpoints.append(corners)
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (9,6), corners2, ret)
                    show(img)
                    cv2.waitKey(1)
            cv2.destroyAllWindows()
            if len(objpoints) > 0 :
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
                mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
                np.save(calib_parameter_path , np.array([[mapx,mapy] , roi]))
                print(f"parameter for camera calibration saved in {calib_parameter_path}")
        calib_parameter = np.load(calib_parameter_path,allow_pickle=True)
        self.maps = calib_parameter[0]
        self.roi = calib_parameter[1]
        
    def run(self,img):
        t = timer()
        dst = cv2.remap(img, self.maps[0], self.maps[1], cv2.INTER_LINEAR)
        x, y, w, h = self.roi
        dst_L = dst[y:y+h, x:x+w]
        dst_L = cv2.resize(dst_L , (img.shape[1],img.shape[0]))
        #t.show()
        return dst_L

    def test(self):
        i = 77
        left_path = f"./videos/vid_{i}/out_L.mp4"
        cap_L = cv2.VideoCapture(left_path)
        #cap_L = cv2.VideoCapture("./results/mine/2022_08_08_19_31_26_1.mp4")
        #cap_R = cv2.VideoCapture("./results/mine/2022_08_08_19_31_26_2.mp4")
        while(cap_L.isOpened()):
            ret , frame_L = cap_L.read()
            if ret:
                
                img_L = self.run(frame_L)
                
                show(frame_L ,label= "before_L")
                show(img_L ,label= "after_L")
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
    def save_img(self , vid_path , img_path):
        cap = cv2.VideoCapture(vid_path)
        count = 0
        while(cap.isOpened()):
            if count%10 != 0:
                count +=1
                ret , frame = cap.read()
                continue
            ret , frame = cap.read()
            if ret:
                count +=1
                show(frame ,label="L")

                length = len(glob.glob(os.path.join(img_path,"*"))) + 1
                cv2.imwrite(os.path.join(img_path , f"{length}.jpg") , frame)
                print(f"{length} saved")
                
                #if cv2.waitKey(1) & 0xff==ord('c') :
                    
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
if __name__ == "__main__":

    #cap_L = cv2.VideoCapture(2)
    #cap_R = cv2.VideoCapture(4)

    #cap_L.set(3,w)
    #cap_L.set(4,h)
    #cap_R.set(3,w)
    #cap_R.set(4,h)
    
    c = calibrate(img_path="chessboards/config_2/*")
    print("init done")
    i = 81
    left_path = f"./videos/vid_{i}/*.MP4"
    vid_path = glob.glob(left_path)[0]
    #c.save_img(vid_path)
    c.test()
