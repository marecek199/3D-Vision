import numpy as np
import cv2 as cv
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry as geometry

def evaluate_homography(H, p, q):
    ''' 
    Evaluate the homography H by calculating the reprojection error from point p to q    
    '''
    # Calculate the reprojection error
    p2q = H @ np.array([[p[0]], [p[1]], [1]])
    
    if abs(p2q[2]) < 1e-10:
        return float('inf')
    p2q /= p2q[-1]    
    
    # Evaluate reprojection error
    error = np.linalg.norm(p2q[:2].flatten() - q)
    return error

def findHomography(src, dst, n_sample, ransac_trial, ransac_threshold):
    '''
    1. Implement RANSAC-based homography estimation here.
    2. Use `geometry.calibrate_DLT_homography` to estimate homography from n_sample points.
    3. Return the best homography and the inlier mask.
    '''
    best_score = -1
    best_model = None
    
    for _ in range(ransac_trial):
        # Step 1: Hypothesis generation
        sample_idx = np.random.choice(range(len(src)), size=n_sample, replace=False)
        model = geometry.calibrate_DLT_homography(src[sample_idx], dst[sample_idx])
        
        # Step 2: Hypothesis evaluation
        score = 0
        for (p, q) in zip(src, dst):
            error = evaluate_homography(model, p, q)
            if error < ransac_threshold:
                score += 1
        if score > best_score:
            best_score = score
            best_model = model

    # Generate the best inlier mask
    best_inlier_mask = np.zeros(len(src), dtype=np.uint8)
    for idx, (p, q) in enumerate(zip(src, dst)):
        error = evaluate_homography(best_model, p, q)
        if error < ransac_threshold:
            best_inlier_mask[idx] = 1

    return best_model, best_inlier_mask

if __name__ == '__main__':
    # Load two images
    try:
        img1 = cv.imread('../data/hill01.jpg')
        img2 = cv.imread('../data/hill02.jpg')
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
    H, inlier_mask = findHomography(pts2, pts1, n_sample=4, ransac_trial=500, ransac_threshold=2.0)
    img_merged = geometry.warpPerspective(img2, H, (img1.shape[1]*2, img1.shape[0]))
    
    # Copy first part of img1 to the merged image
    img_merged[:,:img1.shape[1]] = img1 # Copy img1 to the left side of img_merged
    
    # Show the merged image
    img_matched = cv.drawMatches(img1, keypoints1, img2, keypoints2, match, None, None, None,
                                 matchesMask=inlier_mask.tolist(),
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                 )
    
    merge = np.vstack((np.hstack((img1, img2)), img_matched, img_merged))
    cv.imshow(f'Planar Image Stitching with My RANSAC (score={sum(inlier_mask)})', merge)
    cv.imshow('Matched keypoints', img_matched)
    cv.imshow('Merged image', img_merged)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    