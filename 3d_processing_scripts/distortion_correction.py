import numpy as np
import cv2 as cv

if __name__ == '__main__':

    # The given video and calibration data
    video_file = 'data/chessboard.avi'
    K = np.array([[432.7390364738057, 0, 476.0614994349778],
                [0, 431.2395555913084, 288.7602152621297],
                [0, 0, 1]]) # Derived from `calibrate_camera.py`
    dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194])
    
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), 'Cannot read the given input, ' + video_file
    
    # Run distortion correction
    show_rectify = True
    while True:
        valid, img = video.read()
        if not valid:
            break
        
        # Rectify geometric distortion (Alternative: `cv.undistort()`)
        if show_rectify:
            img = cv.undistort(img, K, dist_coeff, None, None)
            info = "Rectified"
        else:
            info = "Original"
            
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        
        # Show the image and process the key event
        cv.imshow("Geometric Distortion Correction", img)
        key = cv.waitKey(10)
        if key == ord(' '):     # Space: Pause
            key = cv.waitKey()
        if key == 27:           # ESC: Exit
            break
        elif key == ord('\t'):  # Tab: Toggle the mode
            show_rectify = not show_rectify
            info = "Rectified" if show_rectify else "Original"
            
    video.release()
    cv.destroyAllWindows()   
