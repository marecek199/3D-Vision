import numpy as np
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def pack_params(K, R, t):
    """
    Pack parameters for camera calibration.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    rvec = Rotation.from_matrix(R).as_rotvec()
    tvec = t.flatten()
    return np.concatenate(([fx, fy, cx, cy], rvec, tvec))

def unpack_params(params):
    """
    Unpack parameters for camera calibration.
    """
    fx, fy, cx, cy = params[0:4]
    rvec = params[4:7]
    tvec = params[7:10]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R = Rotation.from_rotvec(rvec).as_matrix()
    return K, R, tvec

def pack_params_dist(K, dist_coeffs , R, t):
    """
    Pack parameters for camera calibration including distortion coefficients.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    rvec = Rotation.from_matrix(R).as_rotvec()
    tvec = t.flatten()
    return np.concatenate(([fx, fy, cx, cy], dist_coeffs, rvec, tvec))

def unpack_params_dist(params):
    """
    Unpack parameters including distortion coefficients.
    """
    fx, fy, cx, cy = params[0:4]
    dist_coeffs = params[4:9]
    rvec = params[9:12]
    tvec = params[12:15]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R = Rotation.from_rotvec(rvec).as_matrix()
    return K, R, tvec, dist_coeffs