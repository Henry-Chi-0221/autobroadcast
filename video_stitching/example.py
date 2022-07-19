from pickletools import uint8
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from utils import load_imgs_heic , load_img_kitti
from time import time
import cupy as cp
from cython_simple import coor_load
import sys
from cupyx.scipy.ndimage import binary_dilation
class Stitcher:
    def __init__(self):
        pass
    def stitch(self, imgs, blending_mode = "linearBlending", ratio = 0.75):
        '''
            The main method to stitch image
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        print("Left img size (", hl, "*", wl, ")")
        print("Right img size (", hr, "*", wr, ")")
        
        # Step1 - extract the keypoints and features by SIFT detector and descriptor
        print("Step1 - Extract the keypoints and features by SIFT detector and descriptor...")
        kps_l, features_l = self.detectAndDescribe(img_left)
        kps_r, features_r = self.detectAndDescribe(img_right)
        

        # Step2 - extract the match point with threshold (David Lowe’s ratio test)
        print("Step2 - Extract the match point with threshold (David Lowe’s ratio test)...")
        matches_pos = self.matchKeyPoint(kps_l, kps_r, features_l, features_r, ratio)
       
        print("The number of matching points:", len(matches_pos))
        if len(matches_pos) < 500 :
            print("Too few matched points. Bad sample.")
            sys.exit()
        # Step2 - draw the img with matching point and their connection line
        self.drawMatches([img_left, img_right], matches_pos)
        
        # Step3 - fit the homography model with RANSAC algorithm
        print("Step3 - Fit the best homography model with RANSAC algorithm...")
        HomoMat = self.fitHomoMat(matches_pos)
        
        np.save('best_homomat' , HomoMat)
        print("Homography matrix saved.")
        # Step4 - warp image to create panoramic image
        #print("Step4 - warp image to create panoramic image...")
        #warp_img = self.warp([img_left, img_right], HomoMat, blending_mode) 
        return HomoMat
        #return warp_img
    
    def detectAndDescribe(self, img):
        '''
        The Detector and Descriptor
        '''
        # SIFT detector and descriptor
        sift = cv2.SIFT_create()
        kps, features = sift.detectAndCompute(img,None)
        
        return kps, features
    
    def matchKeyPoint(self, kps_l, kps_r, features_l, features_r, ratio):
        '''
            Match the Keypoints beteewn two image
        '''
        Match_idxAndDist = [] # min corresponding index, min distance, seccond min corresponding index, second min distance
        for i in range(len(features_l)):
            min_IdxDis = [-1, np.inf]  # record the min corresponding index, min distance
            secMin_IdxDis = [-1 ,np.inf]  # record the second corresponding min index, min distance
            for j in range(len(features_r)):
                dist = np.linalg.norm(features_l[i] - features_r[j])
                if (min_IdxDis[1] > dist):
                    secMin_IdxDis = np.copy(min_IdxDis)
                    min_IdxDis = [j , dist]
                elif (secMin_IdxDis[1] > dist and secMin_IdxDis[1] != min_IdxDis[1]):
                    secMin_IdxDis = [j, dist]
            
            Match_idxAndDist.append([min_IdxDis[0], min_IdxDis[1], secMin_IdxDis[0], secMin_IdxDis[1]])

        # ratio test as per Lowe's paper
        goodMatches = []
        for i in range(len(Match_idxAndDist)):
            if (Match_idxAndDist[i][1] <= Match_idxAndDist[i][3] * ratio):
                goodMatches.append((i, Match_idxAndDist[i][0]))
            
        goodMatches_pos = []
        for (idx, correspondingIdx) in goodMatches:
            psA = (int(kps_l[idx].pt[0]), int(kps_l[idx].pt[1]))
            psB = (int(kps_r[correspondingIdx].pt[0]), int(kps_r[correspondingIdx].pt[1]))
            goodMatches_pos.append([psA, psB])
            
        return goodMatches_pos
    
    def drawMatches(self, imgs, matches_pos):
        '''
            Draw the match points img with keypoints and connection line
        '''
        
        # initialize the output visualization image
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        vis = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
        vis[0:hl, 0:wl] = img_left
        vis[0:hr, wl:] = img_right
        
        # Draw the match
        for (img_left_pos, img_right_pos) in matches_pos:

                pos_l = img_left_pos
                pos_r = img_right_pos[0] + wl, img_right_pos[1]
                cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)
                cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)
                cv2.line(vis, pos_l, pos_r, (255, 0, 0), 1)
                
        # return the visualization
        #plt.figure(4)
        #plt.title("img with matching points")
        #plt.imshow(vis[:,:,::-1])
        #cv2.imwrite("Feature matching img/matching.jpg", vis)
        
        return vis
    
    def fitHomoMat(self, matches_pos):
        '''
            Fit the best homography model with RANSAC algorithm - noBlending、linearBlending、linearBlendingWithConstant
        '''
        dstPoints = [] # i.e. left image(destination image)
        srcPoints = [] # i.e. right image(source image) 
        for dstPoint, srcPoint in matches_pos:
            dstPoints.append(list(dstPoint)) 
            srcPoints.append(list(srcPoint))
        dstPoints = np.array(dstPoints)
        srcPoints = np.array(srcPoints)
        
        homography = Homography()
        
        # RANSAC algorithm, selecting the best fit homography
        NumSample = len(matches_pos)
        threshold = 5.0  
        NumIter = 8000
        NumRamdomSubSample = 4
        MaxInlier = 0
        Best_H = None
        
        for run in range(NumIter):
            SubSampleIdx = random.sample(range(NumSample), NumRamdomSubSample) # get the Index of ramdom sampling
            H = homography.solve_homography(srcPoints[SubSampleIdx], dstPoints[SubSampleIdx])
            
            # find the best Homography have the the maximum number of inlier
            NumInlier = 0 
            for i in range(NumSample):
                if i not in SubSampleIdx:
                    concateCoor = np.hstack((srcPoints[i], [1])) # add z-axis as 1
                    dstCoor = H @ concateCoor.T # calculate the coordination after transform to destination img 
                    if dstCoor[2] <= 1e-8: # avoid divide zero number, or too small number cause overflow
                        continue
                    dstCoor = dstCoor / dstCoor[2]
                    if (np.linalg.norm(dstCoor[:2] - dstPoints[i]) < threshold):
                        NumInlier = NumInlier + 1
            if (MaxInlier < NumInlier):
                MaxInlier = NumInlier
                Best_H = H
                
        print("The Number of Maximum Inlier:", MaxInlier)
        
        return Best_H
    
    def warp(self, imgs, HomoMat, blending_mode):
       
        '''
           warp image to create panoramic image
           There are three different blending method - noBlending、linearBlending、linearBlendingWithConstant
        '''
        a_s = time()
        

        
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        
        stitch_img = np.zeros( (max(hl, hr), wl + wr, 3), dtype="int") # create the (stitch)big image accroding the imgs height and width 
        
        if (blending_mode == "noBlending"):
            stitch_img[:hl, :wl] = img_left
            
        img_right = cv2.warpPerspective(img_right, HomoMat, (stitch_img.shape[1] , stitch_img.shape[0])) #0.0027
        ### -v- 0.08173298835754395 s
        gray = cv2.cvtColor(img_right , cv2.COLOR_BGR2GRAY)
        _ , mask_a = cv2.threshold( gray , 0 , 255 , cv2.THRESH_BINARY_INV)
        #mask_a =  cv2.morphologyEx(mask_a, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1) 
          #0.014
        ### -v- 0.0592188835144043 s
      
        if use_gpu:
            with cp.cuda.Device(0):
                mask_a_gpu = cp.asarray(mask_a)
                img_right_gpu = cp.asarray(img_right)
                stitch_img_gpu = cp.asarray(stitch_img)
                mask_a_gpu =  binary_dilation(mask_a_gpu/255).astype(cp.uint8)
                mask_a_gpu = cp.stack((mask_a_gpu,)*3, axis=-1)
                img_right_gpu = img_right_gpu * ( 1 - mask_a_gpu )
                stitch_img_gpu = ( (mask_a_gpu * stitch_img_gpu) + img_right_gpu ) 
                #stitch_img_gpu  = stitch_img_gpu[: , :]
                stitch_img  = cp.asnumpy(stitch_img_gpu.astype(cp.uint8) )
        else:
            mask_a = (mask_a/255).astype(cp.uint8)
            mask_a =  cv2.morphologyEx(mask_a, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1) 
            mask_a = cv2.cvtColor(mask_a , cv2.COLOR_GRAY2BGR)
            img_right = img_right * ( 1 - mask_a )
            stitch_img = ( (mask_a * stitch_img) + img_right ).astype(cp.uint8)
        
        blender = Blender()
        if (blending_mode == "linearBlending"):
            stitch_img = blender.linearBlending([img_left, stitch_img])
        elif (blending_mode == "linearBlendingWithConstant"):
            stitch_img = blender.linearBlendingWithConstantWidth([img_left, stitch_img])
            print("blending...")
        # remove the black border
        #stitch_img = self.removeBlackBorder(stitch_img)
        #print('Python runtime :' ,time() - a_s,'FPS :' ,  1/ (time() - a_s ))
        return stitch_img
    def removeBlackBorder(self, img):
        '''
        Remove img's the black border 
        '''
        h, w = img.shape[:2]
        reduced_h, reduced_w = h, w
        # right to left
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if (np.count_nonzero(img[i, col]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_w = reduced_w - 1
                
        # bottom to top 
        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(reduced_w):
                if (np.count_nonzero(img[row, i]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_h = reduced_h - 1
        
        return img[:reduced_h, :reduced_w]

class Blender:
    def linearBlending(self, imgs):
        '''
        linear Blending(also known as Feathering)
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        
        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
        
        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # Plot the overlap mask
        #plt.figure(21)
        #plt.title("overlap_mask")
        #plt.imshow(overlap_mask.astype(int), cmap="gray")
        
        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        for i in range(hr): 
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        
        linearBlending_img = np.copy(img_right)
        linearBlending_img[:hl, :wl] = np.copy(img_left)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
        
        return linearBlending_img
    def linearBlendingWithConstantWidth(self, imgs):
        '''
        linear Blending with Constat Width, avoiding ghost region
        # you need to determine the size of constant with
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        constant_width = 3 # constant width
        
        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
                    
        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        for i in range(hr):
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            
            # Find the middle line of overlapping regions, and only do linear blending to those regions very close to the middle line.
            middleIdx = int((maxIdx + minIdx) / 2)
            
            # left 
            for j in range(minIdx, middleIdx + 1):
                if (j >= middleIdx - constant_width):
                    alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                else:
                    alpha_mask[i, j] = 1
            # right
            for j in range(middleIdx + 1, maxIdx + 1):
                if (j <= middleIdx + constant_width):
                    alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                else:
                    alpha_mask[i, j] = 0

        
        linearBlendingWithConstantWidth_img = np.copy(img_right)
        linearBlendingWithConstantWidth_img[:hl, :wl] = np.copy(img_left)
        # linear blending with constant width
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
        
        return linearBlendingWithConstantWidth_img

class Homography:
    def solve_homography(self, P, m):
        """
        Solve homography matrix 

        Args:
            P:  Coordinates of the points in the original plane,
            m:  Coordinates of the points in the target plane


        Returns:
            H: Homography matrix 
        """
        try:
            A = []  
            for r in range(len(P)): 
                #print(m[r, 0])
                A.append([-P[r,0], -P[r,1], -1, 0, 0, 0, P[r,0]*m[r,0], P[r,1]*m[r,0], m[r,0]])
                A.append([0, 0, 0, -P[r,0], -P[r,1], -1, P[r,0]*m[r,1], P[r,1]*m[r,1], m[r,1]])

            u, s, vt = np.linalg.svd(A) # Solve s ystem of linear equations Ah = 0 using SVD
            # pick H from last line of vt  
            H = np.reshape(vt[8], (3,3))
            # normalization, let H[2,2] equals to 1
            H = (1/H.item(8)) * H
        except:
            print("Error occur!")

        return H
    
if __name__ == "__main__":
    images = [load_imgs_heic("./images/test/*")]
    # The stitch object to stitch the image    
    #images = load_img_kitti()
    blending_mode = "noBlending" # three mode - noBlending、linearBlending、linearBlendingWithConstant
    stitcher = Stitcher()
    init = False
    use_gpu = True

    if init:
        best_homomat = stitcher.stitch(images[0], blending_mode)
    else:
        best_homomat = np.load('./best_homomat.npy')

    print('start')
    
    #for imgs in images:
    while(1):
        imgs = images[0]
        s = time()
        #print(imgs[0].shape)
        warp_img = stitcher.warp(imgs,best_homomat , blending_mode)
        print('Python runtime :' ,round(time() - s,3),'ms ,','FPS :' , round( 1/ (time() - s ),3) )
        #print('Done, shape : ' , warp_img.shape )
        cv2.imshow('L' , cv2.resize(imgs[0] , (imgs[0].shape[1]//2 ,imgs[0].shape[0]//2 )) )
        cv2.imshow('R' , cv2.resize(imgs[1] , (imgs[1].shape[1]//2 ,imgs[1].shape[0]//2 )))
        cv2.imshow('result' ,cv2.resize(warp_img , (warp_img.shape[1]//2 ,warp_img.shape[0]//2 )) )
        
        cv2.waitKey(1)