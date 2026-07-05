import numpy as np
import cv2 as cv
import sys
import os

if __name__ == '__main__':
    # Load two images
    img1 = cv.imread('data/000180.jpg')
    img2 = cv.imread('data/000177.jpg')
    
    if img1 is None or img2 is None:
        img1 = cv.imread('../data/000180.jpg')
        img2 = cv.imread('../data/000177.jpg')
        
    if img1 is None or img2 is None:
        print("Error: Could not load images. Check the file paths.")
        sys.exit(1)
        
    # Detect and compute features
    fdetector = cv.BRISK_create()
    keypoints1, descriptors1 = fdetector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = fdetector.detectAndCompute(img2, None)
    
    # Match features
    fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
    matches = fmatcher.match(descriptors1, descriptors2)
    
    # Keep only the best 50 matches (sorted by distance, lowest = best)
    matches = sorted(matches, key=lambda m: m.distance)[:100]
    
    # Draw matches
    img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Resize to half of Full HD (960x540)
    img_matches = cv.resize(img_matches, (1440, 810), interpolation=cv.INTER_AREA)
    
    cv.imshow(f'Feature Matches ({len(matches)} matches)', img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()

