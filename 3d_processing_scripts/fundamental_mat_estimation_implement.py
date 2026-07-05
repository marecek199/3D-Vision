import numpy as np
import cv2 as cv

import sys; import os;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry as geometry


if __name__ == "__main__":
    # Load images
    pts0 = np.loadtxt('data/image_formation0.xyz')
    pts1 = np.loadtxt('data/image_formation1.xyz')
    
    # Estimate fundamental matrix using normalized 8-point algorithm
    F = geometry.findFundamentalMat(pts0, pts1)
    cv_F, mask = cv.findFundamentalMat(pts0, pts1, cv.FM_8POINT)
    
    print("Estimated Fundamental Matrix (Custom Implementation):")
    print(F)

    print("\nEstimated Fundamental Matrix (OpenCV):")
    print(cv_F)
    
    # Compare the two fundamental matrices
    print("\nDifference between the two matrices:")
    print(F - cv_F)
