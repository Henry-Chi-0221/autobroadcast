import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import cv2
from time import time

class VideoStitcher:
    def __init__(self, left_video_in_path, right_video_in_path, video_out_path, video_out_width=3840, display=True ,gpu=True):
        # Initialize arguments
        self.left_video_in_path = left_video_in_path
        self.right_video_in_path = right_video_in_path
        self.video_out_path = video_out_path
        self.video_out_width = video_out_width
        self.display = display
        self.save = False
        
        self.saved_homo_matrix = None
        
        self.mask_L = None
        self.mask_R = None

        self.use_gpu = gpu

        self.result = np.zeros((1080,3840,3), np.uint8)
        self.image_a = None
        self.image_b = None
        self.output_shape = None
    def stitch(self, images, ratio=0.7, reproj_thresh=20.0):
        
        (image_b, image_a) = images
        #print(image_a.dtype ,image_b.dtype)
        if self.saved_homo_matrix is None:
            (keypoints_a, features_a) = self.detect_and_extract(image_a)
            (keypoints_b, features_b) = self.detect_and_extract(image_b)

            matched_keypoints = self.match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)

            if matched_keypoints is None:
                return None

            self.saved_homo_matrix = matched_keypoints[1]
        
        self.image_a = image_a
        self.image_b = image_b
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
        print(len(matches))
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

    def run(self , idx):
        path = f"results/vid_{idx}"
        if self.save:
            os.mkdir(path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #out_L = cv2.VideoWriter(f'{path}/out_L.mp4', fourcc, 60.0, (1920,  1080))
            #out_R = cv2.VideoWriter(f'{path}/out_R.mp4', fourcc, 60.0, (1920,  1080))       
            out_res = cv2.VideoWriter(f'{path}/out_res.mp4', fourcc, 60.0, (3840,  1080))  

        left_video = cv2.VideoCapture(self.left_video_in_path)
        right_video = cv2.VideoCapture(self.right_video_in_path)
        #left_video = cv2.VideoCapture(2)
        #right_video = cv2.VideoCapture(1)

        print('[INFO]: {} and {} loaded'.format(self.left_video_in_path.split('/')[-1],
                                                self.right_video_in_path.split('/')[-1]))
        print('[INFO]: Video stitching starting....')

        # Get information about the videos

        n_frames = min(int(left_video.get(cv2.CAP_PROP_FRAME_COUNT)),
                       int(right_video.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = int(left_video.get(cv2.CAP_PROP_FPS))
        count = 0
        while(left_video.isOpened() and right_video.isOpened()):
            
            ok_l, left = left_video.read()
            ok_r, right = right_video.read()
            if ok_l and ok_r:
                s = time()
                stitched_frame = self.stitch([left, right])
                #print(f"{round(time() - s ,3) *1000} ms  ,FPS : { round(1 / (time() - s ) , 3)} , {round(count / n_frames * 100 , 1)} %" )
                if stitched_frame is None:
                    print("[INFO]: Homography could not be computed!")
                    break
                
                if self.save:
                    #out_L.write(left)
                    #out_R.write(right)
                    out_res.write(np.uint8(stitched_frame*255))
                # If the 'q' key was pressed, break from the loop
                if not self.save:
                    if self.display:
                        # Show the output images
                        #cv2.imshow("Result", stitched_frame)
                        cv2.imshow("L" , left )
                        cv2.imshow("R" , right )
                        #print(left.shape , right.shape , stitched_frame.shape)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                count += 1
        if self.save:
            #out_L.release()
            #out_R.release()
            out_res.release()
        cv2.destroyAllWindows()
        print('[INFO]: Video stitching finished')

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

            if self.use_gpu:
                with cp.cuda.Device(0):
                    self.mask_L = cp.asarray(self.mask_L , cp.float64)
                    self.mask_R = cp.asarray(self.mask_R , cp.float64)
            

        if self.use_gpu:
            with cp.cuda.Device(0):
                L_gpu = cp.asarray(L , cp.float64)
                R_gpu = cp.asarray(R , cp.float64)
                L_gpu *= self.mask_L
                R_gpu *= self.mask_R
                R_gpu[:self.mask_shape[0] , :self.mask_shape[1]] += L_gpu
                tensor_data = from_dlpack(R_gpu.toDlpack())
                
                return tensor_data
                print(tensor_data.dtype)
        else:
            
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
        constant = 100
        #mask1 = np.repeat(np.tile(np.linspace(0, 1, mid_line - x ), (img.shape[0], 1))[:, :, np.newaxis], 1, axis=2)[:,:,0]
        mask2 = np.repeat(np.tile(np.linspace(0, 1, constant), (img.shape[0], 1))[:, :, np.newaxis], 1, axis=2)[:,:,0]
        img[:,overlap_mid_line:] = 0
        mask2 *= (img[: , overlap_mid_line-constant:overlap_mid_line])
        img[:,:overlap_mid_line-constant] = 255
        img[:,overlap_mid_line-constant:overlap_mid_line] -= np.uint8(mask2)
        mask_L = cv2.cvtColor( img , cv2.COLOR_GRAY2BGR) 
        mask_R = cv2.cvtColor( 255 - img , cv2.COLOR_GRAY2BGR) 
        return mask_L , mask_R

# Example call to 'VideoStitcher'
"""
for i in range(30 ,45):
    stitcher = VideoStitcher(left_video_in_path=f'../videos/vid_{i}/out_R.mp4',
                         right_video_in_path=f'../videos/vid_{i}/out_L.mp4',
                         video_out_path=f'../videos/vid_{i}/out_res.mp4')
    stitcher.run(i)
"""
if __name__ == "__main__":
    i = 53
    stitcher = VideoStitcher(left_video_in_path=f'../videos/vid_{i}/out_L.mp4',
                            right_video_in_path=f'../videos/vid_{i}/out_R.mp4',
                            video_out_path=f'../videos/vid_{i}/out_res.mp4')
    stitcher.run(i)