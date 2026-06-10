import numpy as np
import cv2 as cv
import time

import sys; import os;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry as geometry, cv_engine.detector

if __name__ == '__main__':
    
    # Load two images
    img1 = cv.imread('data/hill01.jpg')
    img2 = cv.imread('data/hill02.jpg')
        
    if img1 is None or img2 is None:
        print("Error: Could not load images. Check the file paths.")
        sys.exit(1)

    detector_instance = cv_engine.detector.Detector(
        detector_type='ORB', 
        number_of_features=3000)
    
    detector_types = ['ORB',
    'SIFT',
    # 'AKAZE',
    'BRISK']
        
    detector_selected = 0        
        
    while True:
        # Detect feature points
        time_start = time.time()
        
        detector_instance = cv_engine.detector.Detector(
            detector_type = detector_types[detector_selected], 
            number_of_features=3000)

        kp1, kp2, match = detector_instance.run(img1, img2)
                 
        # Show the matched image
        img_merged = cv.drawMatches(img1, kp1, img2, kp2, match, None)
                 
        img1_kps = cv.drawKeypoints(img1, kp1, None)
        img2_kps = cv.drawKeypoints(img2, kp2, None)
        img_merged = np.hstack((img1_kps, img2_kps))
        
        info = detector_instance.info
        cv.putText(img_merged, info, (5, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.imshow('Feature Matching', img_merged)
        
        # Process the key event
        key = cv.waitKey(0)
        if key == 27: # ESC
            break
        elif key == ord('-') or key == ord('_'):
            detector_selected = (detector_selected - 1) % len(detector_types)
        elif key == ord('+') or key == ord('='):
            detector_selected = (detector_selected + 1) % len(detector_types)

    cv.destroyAllWindows()