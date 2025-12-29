import numpy as np
import scipy.linalg

from . import geometry

# def compute_homography_normalized(obj_pts, img_pts):
#     ''' Computes H using DLT with Normalization '''
#     obj_planar = obj_pts[:, :2] # Drop Z
    
#     obj_norm, T_obj = geometry.normalize_points(obj_planar)
#     img_norm, T_img = geometry.normalize_points(img_pts)
    
#     n = obj_pts.shape[0]
#     A = np.zeros((2 * n, 9))
    
#     for i in range(n):
#         X, Y = obj_norm[i]
#         u, v = img_norm[i]
#         A[2*i]   = [-X, -Y, -1,  0,  0,  0, u*X, u*Y, u]
#         A[2*i+1] = [ 0,  0,  0, -X, -Y, -1, v*X, v*Y, v]
    
#     _, _, Vt = np.linalg.svd(A)
#     H_norm = Vt[-1].reshape(3, 3)
    
#     # Denormalize
#     H = np.linalg.inv(T_img) @ H_norm @ T_obj
#     return H / H[2, 2]
    
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
        H = geometry.calibrate_DLT_homography(obj_p, img_p)
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


def decompose_projection_matrix(A, obj_pts):
    ''' Decomposes P = [R | t] using SVD '''
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
    
    R, t = geometry.ensure_depth_positive(R, t, obj_pts)
    
    return R, t


def decompose_projection_matrix_rq(A, obj_pts):
    ''' Decomposes P = K [R | t] using RQ decomposition '''
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
    R, t = geometry.ensure_depth_positive(R, t, obj_pts)
    
    return K, R, t