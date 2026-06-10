import numpy as np
import cv2 as cv
import random

import sys; import os;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry, cv_engine.optimization, cv_engine.detector
import cv_engine.detector as d

def mouse_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param.append((x, y))

def draw_line(img, line, color=(255, 0, 0), thickness=1):
    assert img.ndim >= 2
    
    h, w, *_ = img.shape
    a, b, c = line  # Line: ax + by + c = 0
    if abs(a) > abs(b):
        # Line is more vertical 
        pt1 = (int(c / -a), 0)
        pt2 = (int((b * h + c) / -a), h)
    else:
        # Line is more horizontal
        pt1 = (0, int(c / -b))
        pt2 = (w, int((a * w + c) / -b))
    
    cv.line(img, pt1, pt2, color, thickness)
    
if __name__ == '__main__':
    
    # Load two images
    img1 = cv.imread('data/KITTI07/image_0/000000.png', cv.IMREAD_COLOR)
    img2 = cv.imread('data/KITTI07/image_0/000023.png', cv.IMREAD_COLOR)
    assert (img1 is not None) and (img2 is not None), 'Cannot read the given images'
    
    # Estimate fundamental matrix F from matched keypoints    
    f, cx, cy = 707.0912, 601.8873, 183.1104 # From the KITTI dataset
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    detector = d.Detector(number_of_features=5000)
    kp1, kp2, matches = detector.run(img1, img2, match_ratio_threshold=0.5, max_matches=500)

    # Extract location of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Visualize matches
    img_matches = cv.drawMatches(img1, [cv.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in pts1],
                                img2, [cv.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in pts2],
                                [cv.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(min(50, len(matches)))],
                                None)
    cv.imshow('Matches Visualization', cv.resize(img_matches, (1280, 720)))
    cv.waitKey(0)

    # Estimate the fundamental matrix
    F, inlier_mask = cv_engine.optimization.findFundamentalMat(pts1, pts2,ransac_trial=100)
    
    # Register event handlers and show images
    wnd1, wnd2 = 'Epipolar Line: Image #1', 'Epipolar Line: Image #2'
    img1_pts, img2_pts = [], []
    cv.namedWindow(wnd1)
    cv.namedWindow(wnd2)
    cv.setMouseCallback(wnd1, mouse_event, img1_pts)
    cv.setMouseCallback(wnd2, mouse_event, img2_pts)
    cv.imshow(wnd1, img1)
    cv.imshow(wnd2, img2)

    # Get a point from a image and draw its correponding epipolar line on the other image
    while True:
        if len(img1_pts) > 0:
            for x, y in img1_pts:
                color = (0, 0, 255)
                cv.circle(img1, (x, y), 4, color, -1)
                epipolar_line = (F @ [[x], [y], [1]]).flatten()
                draw_line(img2, epipolar_line, color, 2)
            img1_pts.clear()
        if len(img2_pts) > 0:
            for x, y in img2_pts:
                color = (255, 0, 0)
                cv.circle(img2, (x, y), 4, color, -1)
                epipolar_line = (F.T @ [[x], [y], [1]]).flatten()
                draw_line(img1, epipolar_line, color, 2)
            img2_pts.clear()
        cv.imshow(wnd2, img2)
        cv.imshow(wnd1, img1)
        key = cv.waitKey(10)
        if key == 27: # ESC
            break    

cv.destroyAllWindows()