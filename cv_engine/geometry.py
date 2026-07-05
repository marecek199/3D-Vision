import numpy as np
from . import utilsEngine as ue 
from . import solvers
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from cv_engine.optimization import reprojection_error_multiple_views_dist

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
    
def calibrate_DLT_homography(obj_pts, img_pts):
    ''' Computes H using DLT with Normalization '''
    obj_planar = obj_pts[:, :2] # Drop Z
    img_planar = img_pts[:, :2] # Drop Z
    
    obj_norm, T_obj = normalize_points(obj_planar)
    img_norm, T_img = normalize_points(img_planar)
    
    n = obj_pts.shape[0]
    A = np.zeros((2 * n, 9))
    
    for i in range(n):
        X, Y = obj_norm[i]
        u, v = img_norm[i]
        A[2*i]   = [-X, -Y, -1,  0,  0,  0, u*X, u*Y, u]
        A[2*i+1] = [ 0,  0,  0, -X, -Y, -1, v*X, v*Y, v]
    
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    
    # Denormalize
    H = np.linalg.inv(T_img) @ H_norm @ T_obj
    
    # Normalize so that H[2,2] = 1
    H /= H[2, 2]  # Ensure H[2,2] is 1
    
    # Homography matrix
    return H

def solvePnP(obj_points, img_points, K, dist_coeff):
    
    # Ensure points are in the correct shape
    if(obj_points.shape[1] != 3):
        raise ValueError("obj_points should be Nx3.")
    obj_points = np.asarray(obj_points, dtype=np.float32)
    
    if(img_points.shape[1] != 2):
        raise ValueError("img_points should be Nx2.")
    img_points = np.asarray(img_points, dtype=np.float32)

    images_num = len(obj_points)
           
    # 1. Initialize Intrinsics + Distortion:    
    unknown_init = []

    # 2. Initialize Extrinsics (R, t) 
    for i in range(images_num):
        try:
            K, R, t = calibrate_DLT(obj_points[i], img_points[i], K)
            rvec = Rotation.from_matrix(R).as_rotvec()
            unknown_init.extend(rvec.tolist())
            unknown_init.extend(t.tolist())
        except Exception as e:
            print(f"View {i} extrinsic init failed: {e}")
            unknown_init.extend([0, 0, 0, 0, 0, 1])

    unknown_init = np.array(unknown_init)
       
    # 3. Optimization
    result = least_squares(
        reprojection_error_multiple_views_dist, 
        unknown_init, 
        args=(obj_points, img_points),
        verbose=0,
        ftol=1e-10, 
        xtol=1e-10, 
        gtol=1e-10, 
        max_nfev=1000
    )

    # 4. Extract results   
    rvecs = []
    tvecs = []
    for i in range(images_num):
        idx = i * 6
        rvecs.append(result.x[idx:idx+3])
        tvecs.append(result.x[idx+3:idx+6])

    # 6. Compute final RMS
    residuals = result.fun
    num_points = sum(len(p) for p in img_points)
    rms = np.sqrt(np.sum(residuals**2) / num_points)        
    
    return rms, rvecs, tvecs

def warpPerspective(src, H, dst_size):
    '''
    Backward Mapping: Destination -> Source
    Note: This implementation avoids holes in the output image.
    '''
    
    # Create an output image
    width, height = dst_size
    channels = src.shape[2] if src.ndim > 2 else 1
    dst = np.zeros((height, width, channels), dtype=src.dtype)
    
    # Compute the inverse homography
    H_inv = np.linalg.inv(H)
    
    # Map each pixel from destination to source
    for qy in range(height):
        for qx in range(width):
            # Project destination pixel (qx, qy) to source (px, py)
            p = H_inv @ [qx, qy, 1]
            # Normalize homogeneous coordinates
            px, py = int(p[0]/p[-1] + 0.5), int(p[1]/p[-1] + 0.5)
            # Check bounds and assign pixel value
            if 0 <= px < src.shape[1] and 0 <= py < src.shape[0]:
                dst[qy, qx] = src[py, px]
    
    return dst  


def evaluate_homography(H, p, q):
    ''' 
    Evaluate the homography H by calculating the reprojection error from point p to q    
    '''
    # Calculate the reprojection error
    p2q = H @ np.array([[p[0]], [p[1]], [1]])
    
    if abs(p2q[2]) < 1e-10:
        return float('inf')
    p2q /= p2q[-1]    
    
    # Evaluate reprojection error
    error = np.linalg.norm(p2q[:2].flatten() - q)
    return error

def findHomography(src, dst, n_sample = 4, ransac_trial = 1000, ransac_threshold = 2.0):
    '''
    1. Implement RANSAC-based homography estimation here.
    2. Use `geometry.calibrate_DLT_homography` to estimate homography from n_sample points.
    3. Return the best homography and the inlier mask.
    '''
    best_score = -1
    best_model = None
    
    for _ in range(ransac_trial):
        # Step 1: Hypothesis generation
        sample_idx = np.random.choice(range(len(src)), size=n_sample, replace=False)
        model = calibrate_DLT_homography(src[sample_idx], dst[sample_idx])
        
        # Step 2: Hypothesis evaluation
        score = 0
        for (p, q) in zip(src, dst):
            error = evaluate_homography(model, p, q)
            if error < ransac_threshold:
                score += 1
        if score > best_score:
            best_score = score
            best_model = model

    # Generate the best inlier mask
    best_inlier_mask = np.zeros(len(src), dtype=np.uint8)
    for idx, (p, q) in enumerate(zip(src, dst)):
        error = evaluate_homography(best_model, p, q)
        if error < ransac_threshold:
            best_inlier_mask[idx] = 1

    return best_model, best_inlier_mask

def evaluate_fundamental(F, p, q):
    ''' 
    Evaluate the fundamental matrix F by calculating the distance 
    from point q to the epipolar line generated by p.
    '''
    # Convert to homogeneous coordinates
    p_h = np.array([p[0], p[1], 1])
    q_h = np.array([q[0], q[1], 1])
    
    # 1. Compute Epipolar Line in the second image: l' = F @ p
    # l' = [a, b, c] corresponding to line ax + by + c = 0
    l = F @ p_h 
    
    # 2. Calculate perpendicular distance from point q to line l'
    # Distance = |ax + by + c| / sqrt(a^2 + b^2)
    # Note: dot(q_h, l) is exactly (ax + by + c)
    numerator = np.abs(np.dot(q_h, l))
    denominator = np.sqrt(l[0]**2 + l[1]**2)
    
    # Avoid division by zero (if line is undefined/at infinity)
    if denominator < 1e-10:
        return float('inf')
        
    return numerator / denominator

def findFundamentalMat(pts1, pts2):
    ''' Estimate the Fundamental Matrix using the normalized 8-point algorithm '''

    if (len(pts1) != len(pts2)) or (len(pts1) < 8):
        raise ValueError("There must be at least 8 point correspondences and both sets must have the same number of points.")

    # # Normalize points
    pts1 = pts1[:, :2] # Drop Z
    pts2 = pts2[:, :2] # Drop Z
    pts1, T1 = normalize_points(pts1)
    pts2, T2 = normalize_points(pts2)

    # Direct implementation of the normalized 8-point algorithm
    # # Make homogeneous points
    if (pts1.shape[1] == 2):
        pts1 = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    if (pts2.shape[1] == 2):
        pts2 = np.hstack([pts2, np.ones((pts2.shape[0], 1))])        

    n = pts1.shape[0]
    A = np.zeros((n, 9))
    # Construct matrix A
    for i, (pts_1, pts_2) in enumerate(zip(pts1, pts2)):
        x1, y1, _ = pts_1
        x2, y2, _ = pts_2
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Solve for F using SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    
    # Denormalize
    if ('T2' in globals() or 'T2' in locals()) and ('T1' in globals() or 'T1' in locals()) :
        F = T2.T @ F @ T1
    
    # Normalize so that F[2,2] = 1
    if abs(F[2, 2]) > 1e-10:
        F /= F[2, 2]
    
    return F
