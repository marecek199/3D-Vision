import numpy as np
from . import utilsEngine as ue 
from . import solvers

# ----------------------------------------------------
# Hartley Normalization
# ----------------------------------------------------
def normalize_points(points):
    ''' Center and scale points so centroid is 0 and avg dist is sqrt(2) '''
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    avg_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / avg_dist
    
    T = np.eye(3)
    T[0, 0] = scale
    T[1, 1] = scale
    T[0, 2] = -scale * centroid[0]
    T[1, 2] = -scale * centroid[1]
    
    # Apply normalization
    # Add ones for homogeneous coords
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    
    return pts_norm[:, :2], T

def project(X, K, R, t):
    ''' Project 3D points X into image points using camera matrix K, rotation R and translation t. '''
    # Ensure X is Nx3
    if X.shape[1] != 3: X = X.T
    assert X.shape[1] == 3, "Input X must have three columns representing 3D points." 
    
    X_cam = R @ X.T + t.reshape((3, 1))
    x_h = (K @ X_cam).T
    x_h = x_h / x_h[:,-1].reshape((-1, 1))  # Normalize
    return x_h[:, 0:2]

def project_distorted(X, K, R, t, dist_coeffs):
    ''' Project 3D points X into image points using camera matrix K, rotation R, translation t and lens distortion. '''
    # Ensure X is Nx3
    # 1. Transform to Camera Coordinates
    if X.shape[1] != 3: X = X.T
    X_cam = R @ X.T + t.reshape((3, 1)) # Shape (3, N)
    
    # 2. Normalize (divide by Z)
    # Avoid division by zero
    z = X_cam[2]
    z[z == 0] = 1e-10
    
    x_n = X_cam[0] / z
    y_n = X_cam[1] / z
    
    # 3. Apply Distortion (Brown-Conrady Model)
    k1, k2, p1, p2, k3 = dist_coeffs
    
    r2 = x_n**2 + y_n**2
    r4 = r2**2
    r6 = r2**3
    
    # Radial distortion: (1 + k1*r^2 + k2*r^4 + k3*r^6)
    radial = 1 + k1*r2 + k2*r4 + k3*r6
    
    # Tangential distortion
    # x_tangential = 2*p1*x*y + p2*(r^2 + 2*x^2)
    # y_tangential = p1*(r^2 + 2*y^2) + 2*p2*x*y
    x_tan = 2*p1*x_n*y_n + p2*(r2 + 2*x_n**2)
    y_tan = p1*(r2 + 2*y_n**2) + 2*p2*x_n*y_n
    
    x_distorted = x_n * radial + x_tan
    y_distorted = y_n * radial + y_tan
    
    # 4. Project to Pixel Coordinates (Apply K)
    # u = fx * x_dist + cx
    # v = fy * y_dist + cy
    u = K[0,0] * x_distorted + K[0,2]
    v = K[1,1] * y_distorted + K[1,2]
    
    return np.vstack((u, v)).T

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

def calibrate_DLT(obj_pts, img_pts, K = None) -> (np.ndarray, np.ndarray, np.ndarray):
    # Number of points
    n = obj_pts.shape[0]
    
    # 1. Convert to homogeneous coordinates
    img_pts_h = np.hstack([img_pts, np.ones((img_pts.shape[0], 1))])
    obj_pts_h = np.hstack([obj_pts, np.ones((obj_pts.shape[0], 1))])
    
    # 2. Normalize image points if K is provided
    if K is not None:
        #  Use normalized points
        img_pts_h_norm = img_pts_h @ np.linalg.inv(K).T
        img_pts_h = img_pts_h_norm    
    else:
        img_pts_h = img_pts_h

    # 3. Build matrix A for DLT
    # x = PX  -->  x cross (PX) = 0 [linear equations in P]
    A = np.zeros((2 * n, 12))    
    for idx, img_pts_h_norm_curr in enumerate(img_pts_h):
        A_curr = np.zeros((2, 12))
        # first row
        A_curr[0, 0:4] = -obj_pts_h[idx]  # X row
        A_curr[0, 4:8] = 0
        A_curr[0, 8:12] = img_pts_h_norm_curr[0] * obj_pts_h[idx]
        # second row
        A_curr[1, 0:4] = 0
        A_curr[1, 4:8] = -obj_pts_h[idx]  # Y row
        A_curr[1, 8:12] = img_pts_h_norm_curr[1] * obj_pts_h[idx]
        # Assign to A
        A[2*idx:2*idx+2, :] = A_curr
    
    if K is not None:
        R, t = solvers.decompose_projection_matrix(A, obj_pts)
    else:
        K, R, t = solvers.decompose_projection_matrix_rq(A, obj_pts)
    
    return K, R, t