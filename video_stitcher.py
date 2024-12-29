import numpy as np
import cv2
from time import time
import sys
from calibration import calibrate
from util import remove_black_border
import math
class VideoStitcher : 
    def __init__(self,fullsize ,initial_frame_count = 20):
        # Initialize arguments
        self.saved_homo_matrix = None
        
        self.mask_L = None
        self.mask_R = None

        self.image_a = None
        self.image_b = None
        self.output_shape = None
        
        self.count = 1
        self.initial_frame_count = initial_frame_count
        self.warp_both = True

        self.calibrate = calibrate()
        self.remove_black_border = remove_black_border()
        

    def mat_sqrt(self,M):
        n = 3
        y = M
        z = np.identity(n)
        for i in range(0,n):
            y_new = (y + np.linalg.inv(z)) / 2
            z_new = (z + np.linalg.inv(y)) / 2
            y = y_new
            z = z_new
        #print(y,z)
        
        return [y , z] # sqrt(M) , inv( sqrt(M) ) 
    def train_homography_mat(self,image_a , image_b,ratio=0.5, reproj_thresh=20.0):
        
        (keypoints_a, features_a) = self.detect_and_extract(image_a)
        (keypoints_b, features_b) = self.detect_and_extract(image_b)

        matched_keypoints = self.match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)

        if matched_keypoints is None:
            return None
        
        #matched_keypoints = list(matched_keypoints)
        #matched_keypoints[1] = self.mat_sqrt(matched_keypoints[1])[0]
        
        if self.saved_homo_matrix is None:
            if self.warp_both:
                
                self.saved_homo_matrix = self.mat_sqrt(matched_keypoints[1]) 
                """
                M1 = self.saved_homo_matrix[0]
                M2 = self.saved_homo_matrix[1]
                theta1 = - math.atan2(M1[0,1], M1[0,0]) * 180 / math.pi
                theta2 = - math.atan2(M2[0,1], M2[0,0]) * 180 / math.pi
                print(theta1 ,theta2)
                """
                self.hmat_offset(x_offset = -800 , y_offset = 6)
                
                
            else:
                self.saved_homo_matrix = [matched_keypoints[1]]
        #else:
        #    self.saved_homo_matrix = (self.saved_homo_matrix * self.count + matched_keypoints[1]) / (self.count+1)
            #self.count +=1
        self.count +=1
    def stitch(self, images, ratio=0.5, reproj_thresh=20.0):

        imgs = self.calibrate.run(images)
        (image_b, image_a) = images
        image_b = imgs[0]
        image_a = imgs[1]
        #print(image_a.dtype ,image_b.dtype)
        
        
        #image_a = self.calibrate.run(image_a) 
        #image_b = self.calibrate.run(image_b) 
        
        self.image_a = image_a 
        self.image_b = image_b 

        if self.count < self.initial_frame_count:
            #print(self.count)
            self.train_homography_mat(self.image_a , self.image_b,ratio, reproj_thresh)
            
        if (self.image_a is not None) and (self.output_shape is None):
            self.output_shape = (self.image_a.shape[1] + self.image_b.shape[1], self.image_a.shape[0])
        
        if self.warp_both :
            
            result_R, res_R, pts_R = self.remove_black_border.run(self.image_a, self.saved_homo_matrix[0], self.output_shape)
            result_L, res_L, pts_L = self.remove_black_border.run(self.image_b, self.saved_homo_matrix[1], self.output_shape)
            # pts : (x, y, w, h)
            print(result_R.shape)
            K = np.array([[800,0,(pts_L[0] + pts_L[2])/2],[0,800,(pts_L[1] + pts_L[3])/2],[0,0,1]])
            result_L = cylindricalWarp(result_L , K)[:,:,:3]

            K = np.array([[800,0,(pts_R[0] + pts_R[2])/2],[0,800,(pts_R[1] + pts_R[3])/2],[0,0,1]])
            result_R = cylindricalWarp(result_R , K)[:,:,:3]
            print(result_R.shape)
            cv2.imshow("res_R" , cv2.resize(result_R , (result_R.shape[1]//2 , result_R.shape[0]//2)))
            cv2.imshow("res_L" , cv2.resize(result_L , (result_L.shape[1]//2 , result_L.shape[0]//2)))
            #cv2.waitKey(0)
            result = self.blending_latest(result_L , result_R)
            
            #result = self.blending_new(result_L , result_R) 
        else:
            result_R = cv2.warpPerspective(self.image_a, self.saved_homo_matrix[0], self.output_shape , flags=cv2.INTER_NEAREST )
            result = self.blending(image_b , result_R)
            #result = self.blending(image_b , result_R)
        #self.saved_homo_matrix = None
        return result
    def hmat_offset(self,x_offset , y_offset):
        transform_mat = np.identity(3)
        transform_mat[0][2] = x_offset
        transform_mat[1][2] = y_offset

        #print(self.saved_homo_matrix[0][:,2])
        self.saved_homo_matrix[0] = self.saved_homo_matrix[0] @ transform_mat
        #print(self.saved_homo_matrix[0][:,2])
        transform_mat[0][2] = -x_offset
        transform_mat[1][2] = -y_offset
        #print(self.saved_homo_matrix[1][:,2])
        self.saved_homo_matrix[1] = self.saved_homo_matrix[1] @ transform_mat
        #print(self.saved_homo_matrix[1][:,2])
        #sys.exit()
    @staticmethod
    def detect_and_extract(image):
        # Detect and extract features from the image (DoG keypoint detector and SIFT feature extractor)
        descriptor = cv2.SIFT_create()
        (keypoints, features) = descriptor.detectAndCompute(image, None)

        # Convert the keypoints from KeyPoint objects to numpy arrays
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        # Return a tuple of keypoints and features
        return (keypoints, features)

    @staticmethod
    def match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh):
        # Compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(features_a, features_b, k=2)
        matches = []

        for raw_match in raw_matches:
            # Ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(raw_match) == 2 and raw_match[0].distance < raw_match[1].distance * ratio:
                matches.append((raw_match[0].trainIdx, raw_match[0].queryIdx))
        # Computing a homography requires at least 4 matches
        if len(matches) > 4:
            # Construct the two sets of points
            points_a = np.float32([keypoints_a[i] for (_, i) in matches])
            points_b = np.float32([keypoints_b[i] for (i, _) in matches])

            # Compute the homography between the two sets of points
            (homography_matrix, status) = cv2.findHomography(points_a, points_b, cv2.RANSAC, reproj_thresh)

            # Return the matches, homography matrix and status of each matched point
            return (matches, homography_matrix, status)

        # No homography could be computed
        return None

    @staticmethod
    def draw_matches(image_a, image_b, keypoints_a, keypoints_b, matches, status):
        (height_a, width_a) = image_a.shape[:2]
        (height_b, width_b) = image_b.shape[:2]
        visualisation = np.zeros((max(height_a, height_b), width_a + width_b, 3), dtype="uint8")
        visualisation[0:height_a, 0:width_a] = image_a
        visualisation[0:height_b, width_a:] = image_b
        for ((train_index, query_index), s) in zip(matches, status):
            if s == 1:
                point_a = (int(keypoints_a[query_index][0]), int(keypoints_a[query_index][1]))
                point_b = (int(keypoints_b[train_index][0]) + width_a, int(keypoints_b[train_index][1]))
                cv2.line(visualisation, point_a, point_b, (0, 255, 0), 1)
        return visualisation

    def blending(self, L, R): # 0.056 ms  ,FPS : 18.009
        if self.mask_L is None :
            self.mask_L , self.mask_R = self.masking(R)
            self.mask_L , self.mask_R = self.mask_L[:,:L.shape[1]] , self.mask_R
            self.mask_L , self.mask_R = self.mask_L/255 , self.mask_R/255
            self.mask_shape = self.mask_L.shape
       
        L_masked = L.copy() * self.mask_L
        R_masked = R.copy() * self.mask_R

        R_masked[:self.mask_shape[0] , :self.mask_shape[1]] += L_masked
        
        R_masked = R_masked.astype(np.uint8)
        
        return R_masked

    def blending_new(self , L , R):
        if self.mask_L is None:
            mask_L , mask_R,L_center , R_center ,constant ,x,w= self.mask_new(L,R)
            self.mask_L , self.mask_R ,self.L_center,self.R_center,self.constant,self.x,self.w =  mask_L , mask_R ,L_center , R_center ,constant ,x ,w
        #cv2.imshow("Lm" , self.mask_L)
        cv2.waitKey(0)
        L = np.uint8(L * self.mask_L ) 

        R = np.uint8(R[:,self.x:self.x+self.w,:] * self.mask_R )

        R = R[:,self.R_center:,:]

        L[:,self.L_center-self.constant:self.L_center-self.constant + R.shape[1],:] += R

        return L
    
    def blending_latest(self,L,R):
        bg = np.zeros([L.shape[0],3840,3] ,dtype=np.uint8)
        #L , R =  np.uint8(L * 0.5) ,np.uint8(R * 0.5)
        x_offset = -620
        width = 50
        #x_offset = -750
        #width = 100
        
        mask_L = np.ones([L.shape[0],L.shape[1],3])
        mask_R = np.ones([R.shape[0],R.shape[1],3])
        
        mask_L[:,L.shape[1] + (x_offset//2) - (width //2) : L.shape[1] + (x_offset//2) + (width - width//2),:] = np.repeat(np.tile(np.linspace(1, 0, width), (L.shape[0], 1))[:, :, np.newaxis], 1, axis=2)[:,:,:]
        mask_R[:,-(x_offset//2) - ( width - width //2 ) : -(x_offset//2) + ( width //2 ) ,:] = np.repeat(np.tile(np.linspace(0, 1, width), (R.shape[0], 1))[:, :, np.newaxis], 1, axis=2)[:,:,:]
        
        mask_L[:,L.shape[1] + (x_offset//2) + (width - width//2):,:] = 0
        mask_R[:,:-(x_offset//2) - ( width - width //2 ),:] = 0

        L = np.uint8(L * mask_L)
        R = np.uint8(R * mask_R)

        bg[:,:L.shape[1],:] = L
        bg[:,L.shape[1] + x_offset:L.shape[1] + R.shape[1] + x_offset,:] += R
        
        #cv2.imshow("b" , cv2.resize(bg , (bg.shape[1]//2 , bg.shape[0]//2)) )
        return bg

    def mask_new(self, L, R):

        #cv2.imshow("L" ,cv2.resize(L , (int(self.output_shape[0]//2) , int(self.output_shape[1]//2))) )
        #cv2.imshow("R" ,cv2.resize(R , (int(self.output_shape[0]//2) , int(self.output_shape[1]//2))) )
        constant = 20

        size = self.image_a.shape
        white = np.ones(size,dtype="uint8")*255
        white_R = cv2.warpPerspective(white, self.saved_homo_matrix[0], self.output_shape , flags=cv2.INTER_NEAREST )
        white_L = cv2.warpPerspective(white, self.saved_homo_matrix[1], self.output_shape , flags=cv2.INTER_NEAREST )
        
        L_center , L_border = self.check_center(white_L)

        white_L[: , :L_center-constant] = 255
        white_L[:,L_center-constant:L_center] = np.repeat(np.tile(np.linspace(255, 0, constant), (size[0], 1))[:, :, np.newaxis], 1, axis=2)[:,:,:]
        white_L[: , L_center:] = 0

        x,y,w,h = self.find_max_contour(white_R)
        #cv2.rectangle(white_R, (x,y) , (x+w, y+h) , (0,255,0) , 10)
        white_R = white_R[:,x:x+w,:]
        

        R_center = L_border - L_center
        white_R[: , R_center + constant:] = 255
        white_R[:,R_center:R_center+constant] = np.repeat(np.tile(np.linspace(0, 255, constant), (size[0], 1))[:, :, np.newaxis], 1, axis=2)[:,:,:]
        white_R[: , :R_center] = 0

        
        white_L = white_L /  255
        white_R = white_R /  255

       

        return white_L ,white_R ,L_center , R_center , constant ,x ,w

    def check_center(self,img):
        size = self.image_a.shape
        x,y,w,h = self.find_max_contour(img)
        border = 0
        for i in range(0 , x+w):
            if img[0,i,0] == 0:
                border = i
                break
            if img[size[0]-1,i,0] == 0:
                border = i
                break

        cv2.rectangle(img, (0,y) , (border, y+h) , (0,255,0) , 10)
        center = ( border + (x+w) ) //2
        cv2.rectangle(img, (0,y) , (center, y+h) , (0,0,255) , 10)
        return center , x+w
    def find_max_contour(self , img ,inv = False):
        (cnts, _) = cv2.findContours(img[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            max_area,id = 100 ,0
            for i,c in enumerate(cnts):
                area = cv2.contourArea(c)
                if area > max_area:
                    max_area = area
                    id = i
        (x, y, w, h) = cv2.boundingRect(cnts[id])
        return (x, y, w, h)
    def masking(self,img):
        img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        ret , img = cv2.threshold(img , 0,255  , cv2.THRESH_BINARY)
        (cnts, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            max_area,id = 100 ,0
            for i,c in enumerate(cnts):
                area = cv2.contourArea(c)
                if area > max_area:
                    max_area = area
                    id = i
        (x, y, w, h) = cv2.boundingRect(cnts[id]) 
        mid_line = (img.shape[1]//2)
        overlap_mid_line  = (mid_line + x)//2
        constant = 50
        #mask1 = np.repeat(np.tile(np.linspace(0, 1, mid_line - x ), (img.shape[0], 1))[:, :, np.newaxis], 1, axis=2)[:,:,0]
        mask2 = np.repeat(np.tile(np.linspace(0, 1, constant), (img.shape[0], 1))[:, :, np.newaxis], 1, axis=2)[:,:,0]
        img[:,overlap_mid_line:] = 0
        mask2 *= (img[: , overlap_mid_line-constant:overlap_mid_line])
        img[:,:overlap_mid_line-constant] = 255
        img[:,overlap_mid_line-constant:overlap_mid_line] -= np.uint8(mask2)
        mask_L = cv2.cvtColor( img , cv2.COLOR_GRAY2BGR) 
        mask_R = cv2.cvtColor( 255 - img , cv2.COLOR_GRAY2BGR) 
        return mask_L , mask_R


def cylindricalWarp(img, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA)
  
if __name__ == "__main__":
    from util import recorder
    B = None
    count = 0
    i = 59
    left_path = f'./videos/vid_{i}/out_L.mp4'
    right_path = f'./videos/vid_{i}/out_R.mp4'

    #left_path = "results/mine/2022_08_08_19_31_26_1.mp4"
    #right_path = "results/mine/2022_08_08_19_31_26_2.mp4"
    #left_path = 2
    #right_path = 4

    cap_L = cv2.VideoCapture(left_path)
    cap_R = cv2.VideoCapture(right_path)
    cap_L.set(3 , 1920)
    cap_L.set(4 , 1080)
    cap_R.set(3 , 1920)
    cap_R.set(4 , 1080)
    length = int(cap_L.get(cv2.CAP_PROP_FRAME_COUNT))
    size = int(cap_L.get(3)) , int(cap_R.get(4))
    full_size = int(size[0]*2) , int(size[1])
    stitcher = VideoStitcher(fullsize=full_size , initial_frame_count=2)
    record = False

    if record:
        rec = recorder(full_size , size ,cap_L.get(5)//3)

    while(cap_L.isOpened() and cap_R.isOpened()):
        count +=1

        if count%3 != 0:
            ret_L , frame_L = cap_L.read()
            _     , frame_R = cap_R.read()
            continue
        
        ret,frame_L = cap_L.read()
        _  ,frame_R = cap_R.read()
        
        frame_R = cv2.resize(frame_R , (1920,1080))
        
        if ret:
            s = time()
            
            stitched = stitcher.stitch([frame_L ,frame_R])
            cv2.imshow("frame_L" ,cv2.resize(frame_L , (int(frame_L.shape[1]//2) , int(frame_L.shape[0]//2))) )
            cv2.imshow("frame_R" ,cv2.resize(frame_R , (int(frame_R.shape[1]//2) , int(frame_R.shape[0]//2))) )
            h, w = stitched.shape[:2]
            print(f"{round(time() - s ,3) *1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} ")
            print(round(count / length *100,2) , "%")
            #stitched = stitched[:,:2764,:]
            cv2.imshow("stitched" ,cv2.resize(stitched , (stitched.shape[1]//2 , stitched.shape[0]//2 )) )
                
            if record:
                rec.write(stitched , frame_R)
        else:
            break
        if cv2.waitKey(1) & 0xff==ord('q'):
            break
    if record:
        rec.release()
