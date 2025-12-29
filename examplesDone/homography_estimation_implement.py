import numpy as np
import cv2 as cv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry as geometry


if __name__ == '__main__':
    src = np.array([[115, 401], [776, 180], [330, 793], [1080, 383]], dtype=np.float32)
    dst = np.array([[0, 0], [900, 0], [0, 500], [900, 500]], dtype=np.float32)

    # my_H = geometry.calibrate_DLT(src, dst)
    my_H = geometry.calibrate_DLT_homography(src, dst)
    cv_H = cv.getPerspectiveTransform(src, dst) # Note) It accepts only 4 pairs of points.

    print('\n### My Planar Homography')
    print(my_H)
    print('\n### OpenCV Planar Homography')
    print(cv_H)