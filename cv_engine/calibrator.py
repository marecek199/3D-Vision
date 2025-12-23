import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from . import solvers, optimization

from cv_engine.solvers import init_camera_matrix_zhang, compute_homography_normalized, get_extrinsics_from_homography
from cv_engine.geometry import project, project_distorted, ensure_depth_positive, calibrate_DLT
from cv_engine.optimization import reprojection_error_multiple_views, reprojection_error_multiple_views_dist
import cv_engine.utilsEngine as ue

from dataclasses import dataclass

@dataclass
class CalibrationConfig:
    use_zhang_init: bool = True
    use_homography_init: bool = True
    use_distortion: bool = True
    verbose: int = 1  # 0: Silent, 1: Report, 2: Detailed

class CameraCalibrator:
    def __init__(self, img_size, config: CalibrationConfig = CalibrationConfig()):
        self.img_size = img_size
        self.config = config  # Store the config
        
        # Results storage
        self.K = None
        self.dist_coeff = None
        self.rvecs = None
        self.tvecs = None
        self.rmse = 0.0
        
    def calibrate(self, obj_pts_list, img_pts_list) -> (np.ndarray, np.ndarray, list, list, float):
        if self.config.use_distortion:
            return self._calibrate_with_distortion(obj_pts_list, img_pts_list)
        else:
            return self._calibrate(obj_pts_list, img_pts_list)
        
    def _calibrate_with_distortion(self, obj_pts, img_pts):    
        images_num = len(obj_pts)
        
        fx_init, fy_init, cx_init, cy_init = self._init_intrinsics(self.img_size)
        K_init = np.array([[fx_init, 0, cx_init], [0, fy_init, cy_init], [0, 0, 1]])
        
        # 1. Initialize Intrinsics + Distortion:
        # [fx, fy, cx, cy, k1, k2, p1, p2, rvecs..., tvecs...]
        unknown_init = [fx_init, fy_init, cx_init, cy_init, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 2. Initialize Extrinsics (R, t) 
        for i in range(images_num):
            try:
                K, R, t = calibrate_DLT(obj_pts[i], img_pts[i], K_init)
                rvec = Rotation.from_matrix(R).as_rotvec()
                unknown_init.extend(rvec.tolist())
                unknown_init.extend(t.tolist())
            except Exception as e:
                print(f"View {i} extrinsic init failed: {e}")
                unknown_init.extend([0, 0, 0, 0, 0, 1])

        unknown_init = np.array(unknown_init)
        
        # 3. Setup bounds
        # fx, fy, cx, cy, k1, k2, p1, p2, k3
        lower_bounds = [100, 100, 0, 0, -10, -10, -1, -1, -10] + [-np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf] * images_num
        upper_bounds = [10000, 10000, self.img_size[0], self.img_size[1], 10, 10, 1, 1, 10] + [np.pi, np.pi, np.pi, np.inf, np.inf, np.inf] * images_num
        
        # 4. Optimization
        result = least_squares(
            reprojection_error_multiple_views_dist, 
            unknown_init, 
            args=(obj_pts, img_pts),
            bounds=(lower_bounds, upper_bounds),
            verbose=0,
            ftol=1e-10, 
            xtol=1e-10, 
            gtol=1e-10, 
            max_nfev=1000
        )
    
        # 5. Extract results
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = result.x[0:9]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        
        rvecs = []
        tvecs = []
        for i in range(images_num):
            idx = 9 + i * 6
            rvecs.append(result.x[idx:idx+3])
            tvecs.append(result.x[idx+3:idx+6])

        # 6. Compute final RMS
        residuals = result.fun
        num_points = sum(len(p) for p in img_pts)
        rms = np.sqrt(np.sum(residuals**2) / num_points)        
        
        return rms, K, dist_coeffs, rvecs, tvecs        
        
        
                    
    def _calibrate(self, obj_pts, img_pts):
        images_num = len(obj_pts)
        
        fx_init, fy_init, cx_init, cy_init = self._init_intrinsics(self.img_size)
        
        # 1. Initialize Intrinsics + Distortion:
        # [fx, fy, cx, cy, k1, k2, p1, p2, rvecs..., tvecs...]
        unknown_init = [fx_init, fy_init, cx_init, cy_init, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 2. Initialize Extrinsics (R, t) 
        for i in range(images_num):
            try:
                K, R, t = calibrate_DLT(obj_pts[i], img_pts[i])
                rvec = Rotation.from_matrix(R).as_rotvec()
                unknown_init.extend(rvec.tolist())
                unknown_init.extend(t.tolist())
            except Exception as e:
                print(f"View {i} extrinsic init failed: {e}")
                unknown_init.extend([0, 0, 0, 0, 0, 1])

        unknown_init = np.array(unknown_init)
        
        # 3. Setup bounds
        # fx, fy, cx, cy, k1, k2, p1, p2, k3
        lower_bounds = [100, 100, 0, 0, -10, -10, -1, -1, -10] + [-np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf] * images_num
        upper_bounds = [10000, 10000, self.img_size[0], self.img_size[1], 10, 10, 1, 1, 10] + [np.pi, np.pi, np.pi, np.inf, np.inf, np.inf] * images_num
        
        # 4. Optimization
        result = least_squares(
            reprojection_error_multiple_views_dist, 
            unknown_init, 
            args=(obj_pts, img_pts),
            bounds=(lower_bounds, upper_bounds),
            verbose=0,
            ftol=1e-10, 
            xtol=1e-10, 
            gtol=1e-10, 
            max_nfev=1000
        )
    
        # 5. Extract results
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = result.x[0:9]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        
        rvecs = []
        tvecs = []
        for i in range(images_num):
            idx = 9 + i * 6
            rvecs.append(result.x[idx:idx+3])
            tvecs.append(result.x[idx+3:idx+6])

        # 6. Compute final RMS
        residuals = result.fun
        num_points = sum(len(p) for p in img_pts)
        rms = np.sqrt(np.sum(residuals**2) / num_points)        
        
        return rms, K, dist_coeffs, rvecs, tvecs   
    
    
    def _init_intrinsics(self, img_size):
        fx_init = (img_size[0] + img_size[1]) / 2.0
        fy_init = fx_init
        cx_init = img_size[0] / 2.0
        cy_init = img_size[1] / 2.0
        return (fx_init, fy_init, cx_init, cy_init)
        