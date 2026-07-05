import numpy as np
import cv2 as cv

import sys; import os;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry, cv_engine.optimization

if __name__ == "__main__":
    
    # Load the images
    img1 = cv.imread('data/KITTI07/image_0/000000.png')
    img2 = cv.imread('data/KITTI07/image_0/000023.png')
    assert (img1 is not None) and (img2 is not None), 'Cannot read the given images'

    f, cx, cy = 707.0912, 601.8873, 183.1104 # From the KITTI dataset
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    # Detect ORB keypoints and descriptors
    orb = cv.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Match descriptors using BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract location of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    # Estimate the fundamental matrix
    F, inlier_mask = cv_engine.optimization.findFundamentalMat(pts1, pts2,ransac_trial=100)
    
    print("Estimated Fundamental Matrix:\n", F)
    print(f"Number of inliers = {np.sum(inlier_mask)} out of {len(matches)} matches")
    
    # Extract relative camera pose between two images
    E = K.T @ F @ K
    positive_num, R, t, positive_mask = cv.recoverPose(E, pts1, pts2, K, mask=inlier_mask)
    print(f'* R = {R}')
    print(f'* t = {t}')
    print(f'* The position of Image #2 = {-R.T @ t}') # [-0.57, 0.09, 0.82]
    print(f"* Number of inliers = {np.sum(inlier_mask)} out of {len(matches)} matches")


    # Show the matched images
    img_matched = cv.drawMatches(img1, kp1, img2, kp2, matches, None, None, None,
                                matchesMask=inlier_mask.ravel().tolist()) # Remove `matchesMask` if you want to show all putative matches
    cv.namedWindow('Fundamental Matrix Estimation', cv.WINDOW_NORMAL)
    cv.imshow('Fundamental Matrix Estimation', img_matched)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
    