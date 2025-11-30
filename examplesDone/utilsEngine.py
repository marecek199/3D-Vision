import numpy as np
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def check_linear_dependence(vec1, vec2, tolerance=1e-5):
    """Checks if vec1 and vec2 are linearly dependent."""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    
    # 1. Handle Zero Vectors (If one is zero, they are dependent)
    if np.allclose(vec1, 0, atol=tolerance) or np.allclose(vec2, 0, atol=tolerance):
        return True, 0.0 # Return True, lambda (lambda could be anything if one is zero)

    # 2. Find the non-zero ratio (lambda)
    # Get all ratios where the denominator is not zero
    non_zero_indices = np.abs(vec1) > tolerance
    if not np.any(non_zero_indices):
        # Should be caught by the zero vector check, but for safety
        return False, None
    
    ratios = vec2[non_zero_indices] / vec1[non_zero_indices]
    
    # 3. Check if all calculated ratios are approximately equal
    # We check if the variance/spread of the ratios is near zero.
    if np.std(ratios) < tolerance:
        # Linearly Dependent, return True and the scalar factor
        return True, ratios[0]
    else:
        # Not linearly dependent
        return False, None

def ensure_depth_positive(R, t, obj_pts) -> (np.ndarray, np.ndarray):
    """Ensure that the depth of the first object point is positive."""
    # Ensure depth is positive
    P_cam = R @ obj_pts[0] + t
    if P_cam[2] < 0:
        R = -R
        t = -t
    return R, t

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