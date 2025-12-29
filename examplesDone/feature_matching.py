import numpy as np
import cv2 as cv
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry as geometry

if __name__ == '__main__':
    
    # Load two images
    img1 = cv.imread('../data/hill01.jpg')
    img2 = cv.imread('../data/hill02.jpg')
    
    # Fallback for different working directory
    if img1 is None or img2 is None:
        img1 = cv.imread('data/hill01.jpg')
        img2 = cv.imread('data/hill02.jpg')
        
    if img1 is None or img2 is None:
        print("Error: Could not load images. Check the file paths.")
        sys.exit(1)

    # Instantiate feature detectors and matchers
    features = [
    {'name': 'AKAZE',   'detector': cv.AKAZE_create(),               'matcher': cv.DescriptorMatcher_create('BruteForce-Hamming')},
    {'name': 'BRISK',   'detector': cv.BRISK_create(),               'matcher': cv.DescriptorMatcher_create('BruteForce-Hamming')},
    {'name': 'FAST',    'detector': cv.FastFeatureDetector_create(), 'matcher': None}, # No descriptor
    {'name': 'GFTT',    'detector': cv.GFTTDetector_create(),        'matcher': None}, # No descriptor
    {'name': 'KAZE',    'detector': cv.KAZE_create(),                'matcher': None}, # No descriptor
    {'name': 'MSER',    'detector': cv.MSER_create(),                'matcher': None}, # No descriptor
    {'name': 'ORB',     'detector': cv.ORB_create(),                 'matcher': cv.DescriptorMatcher_create('BruteForce-Hamming')},
    {'name': 'SIFT',    'detector': cv.SIFT_create(),                'matcher': cv.DescriptorMatcher_create('BruteForce')},
    ]
    
    # Initialize control parameters
    f_select = 1  # BRISK
    
    while True:
        # Detect feature points
        time_start = time.time()
        keypoints1 = features[f_select]['detector'].detect(img1)
        keypoints2 = features[f_select]['detector'].detect(img2)
        time_detect = time.time()
        
        if features[f_select]['matcher'] is not None:
            # Extract feature descriptors
            keypoints1, descriptors1 = features[f_select]['detector'].compute(img1, keypoints1)
            keypoints2, descriptors2 = features[f_select]['detector'].compute(img2, keypoints2)
            time_compute = time.time()
            
            # Match the feature descriptors
            match = features[f_select]['matcher'].match(descriptors1, descriptors2)
            time_match = time.time()
            
            # Show the matched image
            img_merged = cv.drawMatches(img1, keypoints1, img2, keypoints2, match, None)
            
        else:
            time_compute = time_detect
            time_match = time_compute
            
            img1_keypts = cv.drawKeypoints(img1, keypoints1, None)
            img2_keypts = cv.drawKeypoints(img2, keypoints2, None)
            img_merged = np.hstack((img1_keypts, img2_keypts))
        
        info = features[f_select]['name'] + f': ({(time_detect-time_start)*1000:.0f} + {(time_compute-time_detect)*1000:.0f} + {(time_match-time_compute)*1000:.0f}) = {(time_match-time_start)*1000:.0f} [msec]'
        cv.putText(img_merged, info, (5, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.imshow('Feature Matching', img_merged)
        
        # Process the key event
        key = cv.waitKey(0)
        if key == 27: # ESC
            break
        elif key == ord('-') or key == ord('_'):
            f_select = (f_select - 1) % len(features)
        elif key == ord('+') or key == ord('='):
            f_select = (f_select + 1) % len(features)
    
    cv.destroyAllWindows()