import numpy as np
import cv2 as cv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cv_engine.calibrator import CameraCalibrator, CalibrationConfig

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    img_select = []
    
    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break
        
        if select_all:
            img_select.append(img)
        else:            
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == ord(' '):             # Space: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):
                    img_select.append(img) # Enter: Select the image
            if key == 27:                  # ESC: Exit (Complete image selection)
                break

    cv.destroyAllWindows()
    return img_select


def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []  # 2D points in image plane
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts.reshape(-1, 2))
    assert len(img_points) > 0
    
    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_pts = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`

    config = CalibrationConfig(use_zhang_init=True, use_homography_init=True, use_distortion=True, verbose=1)
    calibrator = CameraCalibrator(img.shape[1::-1], config)
    
         
    # Calibrate the camera
    rms, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera(obj_pts, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)
    rms1, K1, dist_coeff1, rvecs1, tvecs1 = calibrator.calibrate(obj_pts, img_points)
    
    print(f'OpenCV calibration - RMS: {rms}, K:\n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff}')

    print(f'Custom calibration - RMS: {rms1}, K:\n{K1}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff1}')

    return rms, K, dist_coeff, rvecs, tvecs


if __name__ == '__main__':
    video_file = 'data/chessboard.avi'
    board_pattern = (10, 7)
    board_cellsize = 0.025

    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, 'There are no selected images!'
    final_cost, K, distortion, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('\n\n## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {final_cost}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {distortion}')

