import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import sys; import os;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry, cv_engine.optimization, cv_engine.detector, cv_engine.detector_dnn

if __name__ == '__main__':
    # Define video file and parameters
    # video_file = 'data/blais.mp4'
    video_file = 'data/KITTI07/image_0/%06d.png'  
    min_track_error = 5
    
    # Open a video and get an initial image
    video = cv.VideoCapture(video_file)
    assert video.isOpened()
    
    _, gray_prev = video.read()
    assert gray_prev.size > 0
    if gray_prev.ndim >= 3 and gray_prev.shape[2] > 1:
        gray_prev = cv.cvtColor(gray_prev, cv.COLOR_BGR2GRAY)
    
    detector_instance = cv_engine.detector_dnn.DetectorDnn('SUPERPOINT')
    
    # Run the KLT feature tracker
    while True:
        # Grab an image from the video
        valid, img = video.read()
        if not valid:
            break
        if img.ndim >= 3 and img.shape[2] > 1:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
                                
        kp1, kp2, matches = detector_instance.run(gray_prev, gray)
        
        if len(matches) < 4:
            gray_prev = gray
            continue  # Skip this frame
        
        pts1, pts2 = detector_instance.get_matched_points()
        
        if pts1 is None or pts2 is None or len(pts1) < 4 or len(pts2) < 4:
            gray_prev = gray
            continue
        
        # Estimate the fundamental matrix
        F, inlier_mask = cv_engine.optimization.findFundamentalMat(
            pts1, 
            pts2,
            ransac_trial=5,
            ransac_threshold=3.0
            )
        
        gray_prev = gray
        
        if img.ndim < 3 or img.shape[2] < 3:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        # Show the optical flow on the image
        for i, m in enumerate(matches):
            if inlier_mask[i]:
                pt1 = tuple(np.round(kp1[m[0]]).astype(np.int32))
                pt2 = tuple(np.round(kp2[m[1]]).astype(np.int32))
                # draw line 
                cv.line(img, pt1, pt2, (0, 255, 0))
                # draw point
                cv.circle(img, tuple(pt2), 4, (0, 0, 255), -1)
        # text
        info = f'Number of matches: {np.sum(inlier_mask==1)} / {len(matches)}'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0))
        # show image                        
        cv.imshow('KLT Feature Tracking - DNN', img)
        
        key = cv.waitKey(1)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break
        
    video.release()
    cv.destroyAllWindows()
