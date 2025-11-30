import numpy as np
import scipy.linalg
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import scipy
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

import utilsEngine as ue
import optimizerEngine as oe

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

# ----------------------------------------------------
# PnP and calibration functions
# ----------------------------------------------------

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

def reprojection_error(params, X, x):
    K, R, tvec = ue.unpack_params(params)
    projected_points = project(X, K, R, tvec)
    error = (projected_points - x).flatten()
    return error

def reprojection_error_multiple_views(unknown, Xs, xs):
    # Extract K from first 3 parameters
    fx, fy, cx, cy = unknown[0:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    errors = []
    for idx, (X, x) in enumerate(zip(Xs, xs)):
        # Extract rotation and translation for this view
        offset = 3 + idx * 6
        rvec = unknown[offset:offset+3]
        tvec = unknown[offset+3:offset+6]
        
        # Convert rvec to rotation matrix
        R = Rotation.from_rotvec(rvec).as_matrix()
        
        # Project points
        projected_points = project(X, K, R, tvec)
        
        # Calculate error for this view
        error = (x - projected_points).flatten()
        errors.append(error)
    
    # Concatenate all errors into a single flat array
    return np.concatenate(errors)

def reprojection_error_multiple_views_dist(params, obj_pts_list, img_pts_list):
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = params[0:9]
    
    residuals = []
    for i, (obj_pts, img_pts) in enumerate(zip(obj_pts_list, img_pts_list)):
        idx = 9 + i * 6
        rvec = params[idx:idx+3]
        tvec = params[idx+3:idx+6]
        
        R = Rotation.from_rotvec(rvec).as_matrix()
        
        # Project 3D points to Camera coordinates
        X_cam = (R @ obj_pts.T).T + tvec
        
        # Normalize (x, y)
        z = X_cam[:, 2]
        # Avoid division by zero
        z[np.abs(z) < 1e-6] = 1e-6
        
        x = X_cam[:, 0] / z
        y = X_cam[:, 1] / z
        
        # Apply Distortion
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2**3
        
        radial = (1 + k1*r2 + k2*r4 + k3*r6)
        tangential_x = 2*p1*x*y + p2*(r2 + 2*x**2)
        tangential_y = p1*(r2 + 2*y**2) + 2*p2*x*y
        
        x_dist = x * radial + tangential_x
        y_dist = y * radial + tangential_y
        
        # Project to Pixel coordinates
        u = fx * x_dist + cx
        v = fy * y_dist + cy
        
        proj_pts = np.column_stack([u, v])
        residuals.append((proj_pts - img_pts).flatten())
        
    return np.concatenate(residuals)

def compute_homography_normalized(obj_pts, img_pts):
    ''' Computes H using DLT with Normalization '''
    obj_planar = obj_pts[:, :2] # Drop Z
    
    obj_norm, T_obj = normalize_points(obj_planar)
    img_norm, T_img = normalize_points(img_pts)
    
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
    return H / H[2, 2]

def create_v_ij(h, i, j):
    ''' Helper for Zhang's closed form solution '''
    # Convert indices 1-based to 0-based for python
    i, j = i-1, j-1
    
    v_ij = np.array([
        h[0, i] * h[0, j],
        h[0, i] * h[1, j] + h[1, i] * h[0, j],
        h[1, i] * h[1, j],
        h[2, i] * h[0, j] + h[0, i] * h[2, j],
        h[2, i] * h[1, j] + h[1, i] * h[2, j],
        h[2, i] * h[2, j]
    ])
    return v_ij

def init_camera_matrix_zhang(obj_pts_list, img_pts_list, img_size):
    ''' 
    Solves for K using Zhang's algebraic closed-form solution.
    This avoids guessing 'f' and 'c'.
    '''
    V = []
    homographies = []
    
    for obj_p, img_p in zip(obj_pts_list, img_pts_list):
        H = compute_homography_normalized(obj_p, img_p)
        homographies.append(H)
        
        # Constraints based on orthogonality of rotation columns
        # h1, h2 are columns of H
        # 1. h1.T * B * h2 = 0
        # 2. h1.T * B * h1 = h2.T * B * h2
        v12 = create_v_ij(H, 1, 2)
        v11 = create_v_ij(H, 1, 1)
        v22 = create_v_ij(H, 2, 2)
        
        V.append(v12)
        V.append(v11 - v22)
        
    V = np.array(V)
    
    # Solve Vb = 0 using SVD
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    
    # Construct B matrix (symmetric)
    # B = [b0, b1, b3; 
    #      b1, b2, b4; 
    #      b3, b4, b5]
    B11, B12, B22, B13, B23, B33 = b
    
    # Extract intrinsics from B (Zhang's Appendix B)
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lambda_val = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lambda_val / B11)
    beta = np.sqrt(lambda_val * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_val # Skew
    u0 = gamma * v0 / beta - B13 * alpha**2 / lambda_val
    
    # Reconstruct K
    K_est = np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,     1]
    ])
    
    # Sanity check: if values are negative or nan, fallback to guess
    if np.isnan(K_est).any() or alpha < 0 or beta < 0:
        print("Zhang's algebraic init failed (likely too few views or noise). using fallback.")
        f_guess = (img_size[0] + img_size[1]) / 2
        return np.array([[f_guess, 0, img_size[0]/2], [0, f_guess, img_size[1]/2], [0, 0, 1]]), homographies
        
    return K_est, homographies

def get_extrinsics_from_homography(H, K):
    ''' Recover R, t from Homography '''
    h_norm = np.linalg.inv(K) @ H
    scale = 1.0 / np.linalg.norm(h_norm[:, 0]) # lambda
    
    r1 = h_norm[:, 0] * scale
    r2 = h_norm[:, 1] * scale
    t  = h_norm[:, 2] * scale
    r3 = np.cross(r1, r2)
    
    R_raw = np.column_stack((r1, r2, r3))
    
    # Force proper rotation matrix
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0: R = -R; t = -t
        
    return R, t

def calibrate_DLT(obj_pts, img_pts, K) -> (np.ndarray, np.ndarray, np.ndarray):
    ''' Direct Linear Transform (DLT) method for PnP problem with unknown K '''
    # Number of points
    n = obj_pts.shape[0]
    # Homogeneous coordinates
    img_pts_h = np.hstack([img_pts, np.ones((img_pts.shape[0], 1))])
    obj_pts_h = np.hstack([obj_pts, np.ones((obj_pts.shape[0], 1))])
    
    img_pts_h_norm = img_pts_h @ np.linalg.inv(K).T

    # x = PX  -->  x cross (PX) = 0 [linear equations in P]
    A = np.zeros((2 * n, 12))    
    for idx, img_pts_h_norm_curr in enumerate(img_pts_h_norm):
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
    
    # Solve for P using SVD
    _, _, Vt = np.linalg.svd(A)
    P_vec = Vt[-1] # Last row of Vt
    P = P_vec.reshape(3, 4) # P is 3x4 matrix
    
    # Extract R and t from P
    # P = K [R | t] 
    R_raw = P[:, :3] # 3x3 matrix [R] matrix
    t_raw = P[:, 3] # translation part [t]
    
    # Ensure R is a proper rotation matrix (det(R) = +1)
    U, S, Vt_R = np.linalg.svd(R_raw)
    R = U @ Vt_R
    
    # rotation correction - ensure its not mirror reflection (det(R) = +1)
    if np.linalg.det(R) < 0:
        R = -R
    
    scale = np.mean(S) # Average of singular values as scale factor
    # scale = np.mean(np.divide(R_raw, R)) # Average scale factor
    t = t_raw / scale
    
    R, t = ue.ensure_depth_positive(R, t, obj_pts)
    
    return K, R, t

def calibrate_DLT(obj_pts, img_pts) -> (np.ndarray, np.ndarray, np.ndarray):
    ''' Direct Linear Transform (DLT) method for PnP problem with unknown K '''
    # Number of points
    n = obj_pts.shape[0]    
    # Homogeneous coordinates
    img_pts_h = np.hstack([img_pts, np.ones((img_pts.shape[0], 1))])
    obj_pts_h = np.hstack([obj_pts, np.ones((obj_pts.shape[0], 1))])
    

    # x = PX  -->  x cross (PX) = 0 [linear equations in P]
    A = np.zeros((2 * n, 12))    
    for idx, img_pts_h_curr in enumerate(img_pts_h):
        A_curr = np.zeros((2, 12))
        # first row
        A_curr[0, 0:4] = -obj_pts_h[idx]  # X row
        A_curr[0, 4:8] = 0
        A_curr[0, 8:12] = img_pts_h_curr[0] * obj_pts_h[idx]
        # second row
        A_curr[1, 0:4] = 0
        A_curr[1, 4:8] = -obj_pts_h[idx]  # Y row
        A_curr[1, 8:12] = img_pts_h_curr[1] * obj_pts_h[idx]
        # Assign to A
        A[2*idx:2*idx+2, :] = A_curr
        
    # Solve for P using SVD
    _, _, Vt = np.linalg.svd(A)
    P_vec = Vt[-1] # Last row of Vt
    P = P_vec.reshape(3, 4) # P is 3x4 matrix
        
    # Extract K, R and t from P
    # P = K [R | t] = [KR | Kt]
    M = P[:, :3] # 3x3 matrix [KR] matrix
    p4 = P[:, 3] # translation part [Kt]
    
    K_raw, R_raw = scipy.linalg.rq(M)
    # Ensure positive diagonal for K
    T = np.diag(np.sign(np.diag(K_raw))) # get sign of diagonal elements in matrix
    # if T[2, 2] < 0: 
    #     T[2, 2] = 1
    
    # Inserting Identity matrix *T @ np.linalg.inv(T)*
    # M = K_raw @ (T @ np.linalg.inv(T)) @ R_raw
    K = K_raw @ T # if diagonal is negative, make it positive
    R = np.linalg.inv(T) @ R_raw # adjust R accordingly
    # R = T @ R_raw # also works -> T === np.linalg.inv(T) - since T is diagonal with +/-1 elements only

    # Normalize K
    K = K / K[2, 2]
    
    # p4 = K @ t  =>  t = K_inv @ p4
    t = np.linalg.inv(K) @ p4
    
    # Ensure R is a proper rotation matrix (det(R) = +1)
    # Ensure depth is positive
    R, t = ue.ensure_depth_positive(R, t, obj_pts)
        
    # # Validations    
    assert np.allclose(K[1,0], 0) and np.allclose(K[2,0], 0) and np.allclose(K[2,1], 0), "K is not upper triangular"
    # assert np.isclose(K[0,0], K[1,1], 1e-2), "fx and fy are not equal"
    assert np.all(np.diag(K) > 0), "K has negative focal length or principal point"
        
    assert np.isclose(np.linalg.det(R), 1.0, 1e-5), "Determinant of R is not 1"
    
    assert np.all(np.isfinite(K)), "K contains non-finite values"
    assert np.all(np.isfinite(R)), "R contains non-finite values"
    assert np.all(np.isfinite(t)), "t contains non-finite values"
    
    return K, R, t

def calibrateCamera(obj_pts, img_pts, img_size):
    """Calibrate camera from multiple views."""
    
    # 1. Initialization Strategy
    images_num = len(img_pts)
    
    # Parameter structure: [fx, fy, cx, cy, rvec1, tvec1, rvec2, tvec2, ...]
    # Total: 3 + 6*images_num parameters
    # Better Focal-length initialization
    
    # f_init = (img_size[0] + img_size[1]) / 2.0
    f_guess = (img_size[0] + img_size[1]) / 2.0
    c_guess = [img_size[0]/2.0, img_size[1]/2.0]
    
    # 1. Algebraic Initialization of K
    # K_init, homographies = init_camera_matrix_zhang(obj_pts, img_pts, img_size)
    try:
        K_init, homographies = init_camera_matrix_zhang(obj_pts, img_pts, img_size)
        
        # VALIDATION: Check if Zhang's result makes physical sense
        # A focal length < 100 pixels is impossible for a standard camera (fisheye maybe, but not standard)
        if K_init[0,0] < 100 or K_init[1,1] < 100:
            print(f"Warning: Zhang's Init returned unstable f ({K_init[0,0]:.2f}). Reverting to Geometric Guess.")
            raise ValueError("Unstable Zhang Result")
            
        print(f"Zhang's Init Successful: fx={K_init[0,0]:.2f}, fy={K_init[1,1]:.2f}")
        
    except Exception as e:
        # Fallback if SVD fails or result is garbage
        print(f"Using Geometric Initialization.")
        K_init = np.array([[f_guess, 0, c_guess[0]], 
                           [0, f_guess, c_guess[1]], 
                           [0, 0, 1]])
        
        # If we fallback, we need to compute Homographies manually to get R,t
        homographies = []
        for i in range(images_num):
            H = compute_homography_normalized(obj_pts[i], img_pts[i])
            homographies.append(H)
    
    # Ensure Positive Focal Lengths
    K_init[0,0] = abs(K_init[0,0])
    K_init[1,1] = abs(K_init[1,1])
        
    print(f"Initial focal length: {f_guess:.2f}")
    print(f"Image size: {img_size}")
    
    unknown_init = [K_init[0,0], K_init[1,1], K_init[0,2], K_init[1,2]]
    
    # Initialize rotation and translation for each view
    for i in range(images_num):
        # Use planar homography method for initialization
        R_dlt, t_dlt = get_extrinsics_from_homography(homographies[i], K_init)
        
        rvec_init = Rotation.from_matrix(R_dlt).as_rotvec()
        unknown_init.extend(rvec_init)
        unknown_init.extend(t_dlt)

    
    unknown_init = np.array(unknown_init)
    
    print(f"Optimizing {len(unknown_init)} parameters for {images_num} views...")
    
    # TODO: Implement your own Levenberg-Marquardt optimization !!!
    # My gradient optimizer
    # gradient_optimizer_params = gradient_optimizer(
    #     unknown_init, 
    #     obj_pts, 
    #     img_pts, 
    #     cost_function_multiple_views,
    #     num_iterations=50000  # Increase for multiple views
    # )
    # fx, fy, cx, cy = gradient_optimizer_params[0:4]
    # Extract rotation and translation vectors for each view
    # rvecs = []
    # tvecs = []
    # for i in range(images_num):
    #     offset = 4 + i * 6
    #     rvecs.append(gradient_optimizer_params[offset:offset+3])
    #     tvecs.append(gradient_optimizer_params[offset+3:offset+6])
    # # Calculate final cost
    # final_cost = cost_function_multiple_views(gradient_optimizer_params, obj_pts, img_pts)

    # Bounds: [fx, fy, cx, cy] must be positive + reasonable limits
    # Params structure: [fx, fy, cx, cy, r1, t1...]
    lower_bounds = [50,  50,  0, 0]             + [-np.inf] * (6 * images_num)
    upper_bounds = [50000, 50000, img_size[0], img_size[1]] + [np.inf] * (6 * images_num)
        
    # 2. Optimization
    # Scipy Levenberg-Marquardt optimization
    result = least_squares(
        reprojection_error_multiple_views, 
        unknown_init, 
        args=(obj_pts, img_pts),
        bounds=(lower_bounds, upper_bounds),
        verbose=1,
        x_scale='jac',
        xtol = 1e-12,
        gtol = 1e-12,
        loss='soft_l1',
        )
    
    fx, fy, cx, cy = (result['x'][0:4])    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # 3. Unpack Results
    rvecs = []
    tvecs = []
    for i in range(images_num):
        offset = 4 + i * 6  # Correct offset: 4 camera params + i*6
        rvecs.append(result.x[offset:offset+3])
        tvecs.append(result.x[offset+3:offset+6])
        
    # 4. Calculate RMS
    # final_cost = result['cost']
    residuals = result['fun']  # Final residual vector
    num_observations = sum(len(pts) for pts in img_pts)
    final_cost = np.sqrt(np.sum(residuals**2) / num_observations)
    
    # Validations for the results
    assert np.all(np.isfinite(rvecs)), "rvecs contains non-finite values"
    assert np.all(np.isfinite(tvecs)), "tvecs contains non-finite values"    
    assert fx > 0 and fy > 0, "Focal length must be positive"
    assert cx > 0 and cy > 0, "Principal point coordinates must be positive"
              
    return final_cost, K, np.zeros(5), rvecs, tvecs

def calibrateCameraDist(obj_pts, img_pts, img_size):
    """Calibrate camera with distortion coefficients from multiple views."""
    images_num = len(img_pts)
    
    # 1. Initialize Intrinsics (K) using Zhang's method
    # This is robust for planar objects (chessboards)
    try:
        K_init, homographies = init_camera_matrix_zhang(obj_pts, img_pts, img_size)
        fx_init = K_init[0,0]
        fy_init = K_init[1,1]
        cx_init = K_init[0,2]
        cy_init = K_init[1,2]
        print(f"Zhang's Init Successful: fx={fx_init:.2f}, fy={fy_init:.2f}")
    except Exception as e:
        print(f"Zhang's Init Failed ({e}), using heuristic.")
        fx_init = fy_init = (img_size[0] + img_size[1]) / 2.0
        cx_init = img_size[0] / 2.0
        cy_init = img_size[1] / 2.0
        K_init = np.array([[fx_init, 0, cx_init], [0, fy_init, cy_init], [0, 0, 1]])
        # Fallback: compute homographies individually
        homographies = []
        for i in range(images_num):
            # Assuming Z=0 for planar object, we can compute homography
            # obj_pts is Nx3, take x,y
            H, _ = cv.findHomography(obj_pts[i][:, :2].astype(np.float32), img_pts[i].astype(np.float32))
            homographies.append(H)

    # Parameter vector structure:
    # [fx, fy, cx, cy, k1, k2, p1, p2, k3, rvec_1, tvec_1, ..., rvec_N, tvec_N]
    unknown_init = [fx_init, fy_init, cx_init, cy_init, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # 2. Initialize Extrinsics (R, t) from Homographies
    # Do NOT use DLT for planar objects (Z=0), it is singular/unstable
    for i in range(images_num):
        try:
            R, t = get_extrinsics_from_homography(homographies[i], K_init)
            rvec = Rotation.from_matrix(R).as_rotvec()
            unknown_init.extend(rvec.tolist())
            unknown_init.extend(t.tolist())
        except Exception as e:
            print(f"View {i} extrinsic init failed: {e}")
            unknown_init.extend([0,0,0, 0,0,1])

    unknown_init = np.array(unknown_init)
    
    print(f"Optimizing {len(unknown_init)} parameters for {images_num} views...")

    # Bounds for stability
    # fx, fy, cx, cy, k1, k2, p1, p2, k3
    lower_bounds = [100, 100, 0, 0, -10, -10, -1, -1, -10] + [-np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf] * images_num
    upper_bounds = [10000, 10000, img_size[0], img_size[1], 10, 10, 1, 1, 10] + [np.pi, np.pi, np.pi, np.inf, np.inf, np.inf] * images_num

    # Optimization
    result = least_squares(
        reprojection_error_multiple_views_dist, 
        unknown_init, 
        args=(obj_pts, img_pts),
        bounds=(lower_bounds, upper_bounds),
        verbose=2,
        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=1000
    )

    # Extract results
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = result.x[0:9]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, p1, p2, k3])
    
    rvecs = []
    tvecs = []
    for i in range(images_num):
        idx = 9 + i * 6
        rvecs.append(result.x[idx:idx+3])
        tvecs.append(result.x[idx+3:idx+6])

    # Calculate RMS
    residuals = result.fun
    num_points = sum(len(p) for p in img_pts)
    rms = np.sqrt(np.sum(residuals**2) / num_points)
    
    print(f"Optimization finished: RMS={rms:.6f}")
    
    return rms, K, dist_coeffs, rvecs, tvecs


