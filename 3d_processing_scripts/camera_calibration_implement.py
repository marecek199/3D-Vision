import numpy as np
from scipy.optimize import least_squares

import sys; import os;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cv_engine.calibrator import CameraCalibrator, CalibrationConfig
from cv_engine.geometry import project, project_distorted, ensure_depth_positive, calibrate_DLT
from cv_engine.utilsEngine import load_files_with_fallback

if __name__ == '__main__':
    img_size = (640, 480)

    try:
        img_files = ['image_formation1.xyz', 'image_formation2.xyz']
        [img_pts_1, img_pts_2], data_path = load_files_with_fallback(img_files)
        
        # Extract 2D points from loaded data
        img_pts = [img_pts_1[:, :2], img_pts_2[:, :2]]
        
        # Load box points from the same successful path
        [pts], _ = load_files_with_fallback(['box.xyz'], [data_path])
        
        print(f"Successfully loaded files from: {data_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    obj_pts = [pts] * len(img_pts) # Copy the object point as much as the number of image observation

    config = CalibrationConfig( use_zhang_init=False, use_homography_init=False, use_distortion=False, verbose=1)
    cameraCalib = CameraCalibrator(img_size, config)

    # Calibrate the camera
    # rms, K, _, rvecs, tvecs = cameraCalib.calibrate(obj_pts, img_pts)
    _, K, *_ = cameraCalib.calibrate(obj_pts, img_pts)

    print('\n### Ground Truth')
    print('* f, cx, cy = 1000, 320, 240')
    print('\n### My Calibration')
    print(f'* f, cx, cy = {K[0,0]:.1f}, {K[0,2]:.1f}, {K[1,2]:.1f}')
