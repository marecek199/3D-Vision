import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import cv2 as cv
import time


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from cv_engine.calibrator import CameraCalibrator, CalibrationConfig
from cv_engine.geometry import calibrate_DLT, check_linear_dependence
from cv_engine.utilsEngine import pack_params, unpack_params
from cv_engine.optimization import reprojection_error, gradient_optimizer


if __name__ == '__main__':

    f, cx, cy = 1000., 320., 240.
    try:
        obj_pts = np.loadtxt('data/box.xyz')    
        img_pts = np.loadtxt('data/image_formation1.xyz')[:,:2].copy()
    except Exception as e:
        obj_pts = np.loadtxt('../data/box.xyz')
        img_pts = np.loadtxt('../data/image_formation1.xyz')[:,:2].copy()    

    K_init = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    dist_coeff = np.zeros(4)

    # Estimate camera pose
    # Initial guess using DLT method
    K_dlt, R_dlt, t_dlt = calibrate_DLT(obj_pts, img_pts)
    if K_dlt is None:
        print("DLT zlyhalo.")
        exit()
    # R_dlt = Rotation.from_rotvec(rvec_dlt.flatten()).as_matrix()
    init_ori = np.rad2deg(Rotation.from_matrix(R_dlt.T).as_euler('xyz'))
    init_pos = -R_dlt.T @ t_dlt.flatten()
    
    # Refine using gradient descent
    initial_params = pack_params(K_dlt, R_dlt, t_dlt)
    start_time = time.time()

    optimized_params = gradient_optimizer(initial_params, obj_pts, img_pts)
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.4f} seconds.")
    
    K, R, t = unpack_params(optimized_params)
    my_ori = np.rad2deg(Rotation.from_matrix(R.T).as_euler('xyz'))
    my_pos = -R.T @ t.flatten()

    print("\n--- Beží Scipy least_squares (Levenberg-Marquardt)... ---")
    start_time = time.time()
    result_scipy = least_squares(
        reprojection_error, 
        initial_params, 
        args=(obj_pts, img_pts),
        method='lm',
        verbose=0 
    )
    optimized_params_scipy = result_scipy.x
    K_scipy, R_scipy, t_scipy = unpack_params(optimized_params_scipy)
    time_scipy = time.time() - start_time
    print(f"Scipy least_squares completed in {time_scipy:.4f} seconds.")
    scipy_ori = np.rad2deg(Rotation.from_matrix(R_scipy.T).as_euler('xyz'))
    scipy_pos = -R_scipy.T @ t_scipy.flatten()

    # Estimate camera pose using OpenCV
    _, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist_coeff)
    R_cv = Rotation.from_rotvec(rvec.flatten()).as_matrix()
    cv_ori = np.rad2deg(Rotation.from_matrix(R_cv.T).as_euler('xyz'))
    cv_pos = -R_cv.T @ tvec.flatten()

    print('\n### Ground Truth')
    print('* Camera orientation: [-15, 15, 0] [deg]')
    print('* Camera position   : [-2, -2, 0] [m]')
    print('\n### Init Camera Pose')
    print(f'* Camera orientation: {init_ori} [deg]')
    print(f'* Camera position   : {init_pos} [m]')
    print('\n### My Camera Pose')
    print(f'* Camera orientation: {my_ori} [deg]')
    print(f'* Camera position   : {my_pos} [m]')
    print('\n### Scipy Camera Pose')
    print(f'* Camera orientation: {scipy_ori} [deg]')
    print(f'* Camera position   : {scipy_pos} [m]')
    print('\n### OpenCV Camera Pose')
    print(f'* Camera orientation: {cv_ori} [deg]')
    print(f'* Camera position   : {cv_pos} [m]')
    # Check if cv_pos and my_pos are linearly dependent
    dep, lam = check_linear_dependence(cv_pos, my_pos, tolerance=1e-2)
    if dep:
        print(f'\n### The position vectors are linearly dependent with scale factor: {lam:.4f}')
    else:
        print('\n### The position vectors are NOT linearly dependent')