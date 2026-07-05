import numpy as np
import cv2 as cv
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry as geometry

if __name__ == '__main__':
    # Load two images
    try:
        img1 = cv.imread('data/hill01.jpg')
        img2 = cv.imread('data/hill02.jpg')
    except:
        img1 = cv.imread('data/hill01.jpg')
        img2 = cv.imread('data/hill02.jpg')
    
    # Fallback for different working directory
    if img1 is None or img2 is None:
        img1 = cv.imread('data/hill01.jpg')
        img2 = cv.imread('data/hill02.jpg')
        
    if img1 is None or img2 is None:
        print("Error: Could not load images. Check the file paths.")
        sys.exit(1)
        
        
    # Retrieve matching points
    fdetector = cv.BRISK_create()
    keypoints1, descriptors1 = fdetector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = fdetector.detectAndCompute(img2, None)
    
    fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
    match = fmatcher.match(descriptors1, descriptors2)
    
    # Calculate planar homography and merge them
    pts1, pts2 = [], []
    for i in range(len(match)):
        # Save the matched keypoints
        pts1.append(keypoints1[match[i].queryIdx].pt)
        pts2.append(keypoints2[match[i].trainIdx].pt)
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    
    # Calculate homography using RANSAC
    # log(1 - 0.999) / log(1 - 0.3^4) = 849
    H, inlier_mask = geometry.findHomography(pts2, pts1, n_sample=4, ransac_trial=500, ransac_threshold=2.0)
    img_merged = geometry.warpPerspective(img2, H, (img1.shape[1]*2, img1.shape[0]))
    
    # Copy first part of img1 to the merged image
    img_merged[:,:img1.shape[1]] = img1 # Copy img1 to the left side of img_merged
    
    # Show the merged image
    img_matched = cv.drawMatches(img1, keypoints1, img2, keypoints2, match, None, None, None,
                                 matchesMask=inlier_mask.tolist(),
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                 )
    
    merge = np.vstack((np.hstack((img1, img2)), img_matched, img_merged))
    print(f'Planar Image Stitching with My RANSAC (score={sum(inlier_mask)})')
    
    cv.imshow(f'Planar Image Stitching with My RANSAC (score={sum(inlier_mask)})', merge)
    cv.imshow('Matched keypoints', img_matched)
    cv.imshow('Merged image', img_merged)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    