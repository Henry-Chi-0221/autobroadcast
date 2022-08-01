import numpy as np
import cv2
from time import time

class VideoStitcher:
    def __init__(self,fullsize ,initial_frame_count = 20):
        # Initialize arguments
        self.saved_homo_matrix = None
        
        self.mask_L = None
        self.mask_R = None

        self.result = np.zeros((fullsize[1],fullsize[0],3), np.uint8)
        self.image_a = None
        self.image_b = None
        self.output_shape = None
        
        self.count = 1
        self.initial_frame_count = initial_frame_count
    def train_homography_mat(self,image_a , image_b,ratio=0.7, reproj_thresh=20.0):
        
        (keypoints_a, features_a) = self.detect_and_extract(image_a)
        (keypoints_b, features_b) = self.detect_and_extract(image_b)

        matched_keypoints = self.match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)

        if matched_keypoints is None:
            return None

        if self.saved_homo_matrix is None:
            self.saved_homo_matrix = matched_keypoints[1]
        else:
            self.saved_homo_matrix = (self.saved_homo_matrix * self.count + matched_keypoints[1]) / (self.count+1)
            self.count +=1
    def stitch(self, images, ratio=0.7, reproj_thresh=20.0):
        (image_b, image_a) = images
        #print(image_a.dtype ,image_b.dtype)
        
        self.image_a = image_a
        self.image_b = image_b
        if self.count < self.initial_frame_count :
            #print(self.count)
            self.train_homography_mat(image_a , image_b,ratio, reproj_thresh)
        

        if (self.image_a is not None) and (self.output_shape is None):
            self.output_shape = (self.image_a.shape[1] + self.image_b.shape[1], self.image_a.shape[0])
        
        cv2.warpPerspective(self.image_a, self.saved_homo_matrix, self.output_shape , dst = self.result, flags=cv2.INTER_NEAREST) # 0.006200075149536133 ms
        s = time()
        self.result = self.blending(self.image_b , self.result) #0.06116604804992676 ms

        #print(f"{round(time() - s ,3) *1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} ")
        #print(self.result)
        return self.result

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
        s = time()
        if self.mask_L is None :
            self.mask_L , self.mask_R = self.masking(R)
            #self.mask_L , self.mask_R = np.float64(self.mask_L) , np.float64(self.mask_R) 
            #self.mask_L , self.mask_R = self.mask_L[:,:L.shape[1]]/255/255 , self.mask_R/255/255
            self.mask_L , self.mask_R = self.mask_L[:,:L.shape[1]] , self.mask_R

            #print(self.mask_L.dtype,self.mask_R.dtype)

            #sys.exit()

            #cv2.imshow('mask_l' , self.mask_L)
            #self.mask_L , self.mask_R =  self.mask_L.astype(np.uint8) , self.mask_R.astype(np.uint8) 
            self.mask_shape = self.mask_L.shape

        s = time()
        L = cv2.multiply(L,self.mask_L , dtype=cv2.CV_32S )
        R = cv2.multiply(R,self.mask_R , dtype=cv2.CV_32S )
        
        R[:self.mask_shape[0] , :self.mask_shape[1]] += L
        
        #R = cv2.divide(R , 255*np.ones(R.shape) , dtype=cv2.CV_8UC3) #25ms
        R = (R/255).astype(np.uint8)
        #print(f"{round(time() - s ,3) *1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} ")
            
        return R

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

if __name__ == "__main__":
    stitcher = VideoStitcher()
    