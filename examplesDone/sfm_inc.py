from pickletools import read_uint1
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from copy import deepcopy
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def update_camera_pose(cam_vec, R, t):
    """Update camera parameters with new pose (R, t)"""
    t = t.squeeze()
    rvec = Rotation.from_matrix(R).as_rotvec()
    result = np.array([cam_vec[0], cam_vec[1], cam_vec[2], rvec[0], rvec[1], rvec[2], t[0], t[1], t[2]], dtype=np.float32)
    return result

def get_camera_mat(cam_vec):
    """Extract intrinsic matrix K from camera parameters"""
    f, cx, cy = cam_vec[0], cam_vec[1], cam_vec[2]
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

def get_projection_mat(cam_vec):
    """Get projection matrix P = K[R|t] from camera parameters"""    
    K = get_camera_mat(cam_vec)
    R = Rotation.from_rotvec(cam_vec[3:6]).as_matrix()
    t = cam_vec[6:9].reshape(3, 1)
    Rt = np.hstack((R, t))
    return K @ Rt

def isBadPoint(point_3d, camera1, camera2, Z_limit, max_cos_parallax):
    """Validate 3D point based on depth, camera visibility, and parallax"""    
    if point_3d[2] < -Z_limit or point_3d[2] > Z_limit:
        return True
    
    rvec1 = np.array([camera1[3], camera1[4], camera1[5]], dtype=np.float32)
    rvec2 = np.array([camera2[3], camera2[4], camera2[5]], dtype=np.float32)
    R1 = Rotation.from_rotvec(rvec1).as_matrix()
    R2 = Rotation.from_rotvec(rvec2).as_matrix()
    
    t1 = np.array([[camera1[6], camera1[7], camera1[8]]], dtype=np.float32)
    t2 = np.array([[camera2[6], camera2[7], camera2[8]]], dtype=np.float32)
    
    p1 = R1 @ point_3d[:, np.newaxis] + t1.T
    p2 = R2 @ point_3d[:, np.newaxis] + t2.T
    
    if p1[2, 0] <= 0 or p2[2, 0] <= 0:
        return True
    
    v2 = R1 @ R2.T @ p2
    cos_parallax = p1.T @ v2 / (np.linalg.norm(p1) * np.linalg.norm(v2))
    
    if cos_parallax > max_cos_parallax:
        return True
    
    return False

# ============================================================================
# HELPER FUNCTIONS FOR INCREMENTAL REGISTRATION
# ============================================================================

def find_best_unregistered_camera(match_pair, match_inlier, registered_cameras, failed_cameras, img_set):
    """Find unregistered camera that matches most with registered cameras"""
    best_next_cam = -1
    best_match_count = 0

    for cam_idx in range(len(img_set)):
        # Skip if it is already registered OR if it previously failed in this state
        if cam_idx in registered_cameras or cam_idx in failed_cameras:
            continue

        match_count = 0
        for pair_idx, (cam0, cam1) in enumerate(match_pair):
            if (cam_idx == cam0 and cam1 in registered_cameras) or \
               (cam_idx == cam1 and cam0 in registered_cameras):
                match_count += len(match_inlier[pair_idx])

        if match_count > best_match_count:
            best_match_count = match_count
            best_next_cam = cam_idx

    return best_next_cam

def build_keypoint_to_3d_map(registered_cam, points_3d, point_visibility):
    """Build mapping from keypoint indices to 3D point indices"""
    kp_to_3d = {}
    for pt_idx in range(len(points_3d)):
        kp_idx = point_visibility[pt_idx, registered_cam]
        if kp_idx != -1:  # -1 means not visible
            kp_to_3d[kp_idx] = pt_idx
    return kp_to_3d

def get_3d_2d_correspondences_robust(next_cam, registered_cameras, match_pair, match_inlier, 
                                     img_keypoints, img_descriptors, points_3d, 
                                     point_visibility):
    object_pts = []
    image_pts = []
    point_3d_indices = []
    next_kp_indices = [] # Add this array
    confidence_scores = []
    
    # Build keypoint-to-3D mappings
    kp_to_3d_maps = {}
    for reg_cam in registered_cameras:
        # Use the updated signature
        kp_to_3d_maps[reg_cam] = build_keypoint_to_3d_map(reg_cam, points_3d, point_visibility)
    
    for pair_idx, (cam0, cam1) in enumerate(match_pair):
        if next_cam not in [cam0, cam1]:
            continue
        
        other_cam = cam0 if (cam1 == next_cam) else cam1
        if other_cam not in registered_cameras:
            continue
        
        is_cam0_next = (cam0 == next_cam)
        
        for match in match_inlier[pair_idx]:
            if is_cam0_next:
                next_kp_idx = match.queryIdx
                other_kp_idx = match.trainIdx
            else:
                next_kp_idx = match.trainIdx
                other_kp_idx = match.queryIdx
            
            if other_kp_idx not in kp_to_3d_maps[other_cam]:
                continue
            
            pt_3d_idx = kp_to_3d_maps[other_cam][other_kp_idx]
            
            # Check if already observed
            if point_visibility[pt_3d_idx, next_cam] != -1:
                continue
            
            pt_2d = img_keypoints[next_cam][next_kp_idx].pt
            confidence = 1.0 # (Keep your existing descriptor logic here...)
            
            object_pts.append(points_3d[pt_3d_idx])
            image_pts.append(pt_2d)
            point_3d_indices.append(pt_3d_idx)
            next_kp_indices.append(next_kp_idx) # Track the keypoint index
            confidence_scores.append(confidence)
    
    if len(object_pts) > 0:
        confidence_scores = np.array(confidence_scores)
        object_pts = np.array(object_pts, dtype=np.float32)
        image_pts = np.array(image_pts, dtype=np.float32)
        point_3d_indices = np.array(point_3d_indices, dtype=int)
        next_kp_indices = np.array(next_kp_indices, dtype=int)
        
        valid_mask = confidence_scores >= 0.3
        
        object_pts = object_pts[valid_mask]
        image_pts = image_pts[valid_mask]
        point_3d_indices = point_3d_indices[valid_mask]
        next_kp_indices = next_kp_indices[valid_mask] # Apply mask
        confidence_scores = confidence_scores[valid_mask]
    else:
        object_pts = np.array([], dtype=np.float32).reshape(0, 3)
        image_pts = np.array([], dtype=np.float32).reshape(0, 2)
        point_3d_indices = np.array([], dtype=int)
        next_kp_indices = np.array([], dtype=int)
        confidence_scores = np.array([], dtype=np.float32)
    
    return object_pts, image_pts, point_3d_indices, next_kp_indices, confidence_scores

def estimate_pose_with_PnP(next_cam, object_pts, image_pts, cameras, f_init, cx_init, cy_init):
    """Estimate camera pose using PnP with RANSAC"""
    if len(object_pts) < 4:
        return False, None, None, None

    K = np.array([[f_init, 0, cx_init], [0, f_init, cy_init], [0, 0, 1]], dtype=np.float32)
    dist_coeff = np.zeros(5)

    # PnP with RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_pts, image_pts, K, dist_coeff,
        useExtrinsicGuess=False,
        iterationsCount=500,
        reprojectionError=2.0,
        confidence=0.99
    )

    if not success or inliers is None or len(inliers) < 4:
        return False, None, None, None

    # Refine with all inliers
    _, rvec_refined, tvec_refined = cv2.solvePnP(
        object_pts[inliers], image_pts[inliers], K, dist_coeff,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    R, _ = cv2.Rodrigues(rvec_refined)
    
    return True, inliers, R, tvec_refined

def triangulate_new_3d_points(next_cam, registered_cameras, match_pair, match_inlier, 
                             img_keypoints, cameras, points_3d, point_visibility, 
                             Z_limit, max_cos_parallax):
    new_points_3d = []
    new_point_visibility = []

    for pair_idx, (cam0, cam1) in enumerate(match_pair):
        if next_cam not in [cam0, cam1]:
            continue
        other_cam = cam0 if (cam1 == next_cam) else cam1
        if other_cam not in registered_cameras:
            continue
        is_cam0_new = (cam0 == next_cam)

        for match in match_inlier[pair_idx]:
            # (Keep your existing check for "already_in_cloud" here...)

            if is_cam0_new:
                pt_2d_new = img_keypoints[cam0][match.queryIdx].pt
                pt_2d_other = img_keypoints[cam1][match.trainIdx].pt
                kp_new = match.queryIdx
                kp_other = match.trainIdx
            else:
                pt_2d_new = img_keypoints[cam1][match.trainIdx].pt
                pt_2d_other = img_keypoints[cam0][match.queryIdx].pt
                kp_new = match.trainIdx
                kp_other = match.queryIdx

            P_new = get_projection_mat(cameras[next_cam])
            P_other = get_projection_mat(cameras[other_cam])

            pts_4d = cv2.triangulatePoints(P_other, P_new, np.array([pt_2d_other]).T, np.array([pt_2d_new]).T)
            pt_3d = pts_4d[:3, 0] / pts_4d[3, 0]

            if isBadPoint(pt_3d, cameras[other_cam], cameras[next_cam], Z_limit, max_cos_parallax):
                continue

            new_points_3d.append(pt_3d)
            visibility = np.full(len(cameras), -1, dtype=int) # Init with -1
            visibility[other_cam] = kp_other
            visibility[next_cam] = kp_new
            new_point_visibility.append(visibility)

    if new_points_3d:
        return np.array(new_points_3d), np.array(new_point_visibility)
    else:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, len(cameras))
    
def refine_camera_pose(next_cam, cameras, object_pts, image_pts, inlier_indices, 
                       f_init, cx_init, cy_init):
    """Refine camera pose with local bundle adjustment"""
    if inlier_indices is None or len(inlier_indices) < 4:
        return cameras[next_cam]

    K = np.array([[f_init, 0, cx_init], [0, f_init, cy_init], [0, 0, 1]], dtype=np.float32)
    dist_coeff = np.zeros(5)

    rvec_init = Rotation.from_rotvec(cameras[next_cam][3:6]).as_rotvec()
    tvec_init = cameras[next_cam][6:9].reshape(3, 1)

    _, rvec_refined, tvec_refined = cv2.solvePnP(
        object_pts[inlier_indices], image_pts[inlier_indices], K, dist_coeff,
        rvec=rvec_init, tvec=tvec_init, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    R_refined, _ = cv2.Rodrigues(rvec_refined)
    refined_cam = update_camera_pose(cameras[next_cam], R_refined, tvec_refined)
    
    return refined_cam

# ============================================================================
# GLOBAL BUNDLE ADJUSTMENT
# ============================================================================

def collect_observations(registered_cameras, points_3d, point_visibility, img_keypoints):
    """Gather (camera, point, 2D) observations from the visibility matrix."""
    cam_ids = sorted(registered_cameras)
    cam_local = {c: i for i, c in enumerate(cam_ids)}   # global cam id -> 0..n_cam-1
    cam_idx, pt_idx_arr, pts_2d = [], [], []
    for pt_idx in range(len(points_3d)):
        for cam_id in cam_ids:
            kp = point_visibility[pt_idx, cam_id]
            if kp != -1:
                cam_idx.append(cam_local[cam_id])
                pt_idx_arr.append(pt_idx)
                pts_2d.append(img_keypoints[cam_id][kp].pt)
    return (cam_ids,
            np.array(cam_idx, dtype=int),
            np.array(pt_idx_arr, dtype=int),
            np.array(pts_2d, dtype=np.float64))

def _project(points_3d, cam_ext, f, cx, cy):
    """Vectorized projection. cam_ext = [rvec(3), tvec(3)] per row; f/cx/cy per-observation."""
    rvec, tvec = cam_ext[:, :3], cam_ext[:, 3:]
    theta = np.linalg.norm(rvec, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        v = np.nan_to_num(rvec / theta)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dot = np.sum(points_3d * v, axis=1, keepdims=True)
    p_cam = (cos_t * points_3d + sin_t * np.cross(v, points_3d)
             + (1 - cos_t) * dot * v) + tvec              # R @ X + t
    x = p_cam[:, 0] / p_cam[:, 2]
    y = p_cam[:, 1] / p_cam[:, 2]
    return np.stack([f * x + cx, f * y + cy], axis=1).ravel()

def run_global_bundle_adjustment(cameras, points_3d, point_visibility,
                                 registered_cameras, img_keypoints, fix_first=True):
    """Joint optimization of all registered extrinsics + all 3D points."""
    cam_ids, cam_i, pt_i, pts_2d = collect_observations(
        registered_cameras, points_3d, point_visibility, img_keypoints)
    n_cam, n_pts, n_obs = len(cam_ids), len(points_3d), len(pts_2d)
    if n_cam < 2 or n_obs == 0:
        return cameras, points_3d

    n_fixed  = 1 if fix_first else 0            # anchor first camera -> fixes gauge
    n_optcam = n_cam - n_fixed

    # per-observation intrinsics (kept fixed here)
    f_arr  = np.array([cameras[c][0] for c in cam_ids])[cam_i]
    cx_arr = np.array([cameras[c][1] for c in cam_ids])[cam_i]
    cy_arr = np.array([cameras[c][2] for c in cam_ids])[cam_i]

    ext0 = np.array([cameras[c][3:9] for c in cam_ids], dtype=np.float64)
    pts0 = points_3d.astype(np.float64)
    x0 = np.hstack([ext0[n_fixed:].ravel(), pts0.ravel()])
    obs = pts_2d.ravel()

    def residual(x):
        ext = ext0.copy()
        ext[n_fixed:] = x[:n_optcam * 6].reshape(-1, 6)
        pts = x[n_optcam * 6:].reshape(-1, 3)
        return _project(pts[pt_i], ext[cam_i], f_arr, cx_arr, cy_arr) - obs

    # sparsity pattern
    A = lil_matrix((2 * n_obs, n_optcam * 6 + n_pts * 3), dtype=int)
    i = np.arange(n_obs)
    free = cam_i >= n_fixed
    for s in range(6):
        cols = (cam_i[free] - n_fixed) * 6 + s
        A[2 * i[free],     cols] = 1
        A[2 * i[free] + 1, cols] = 1
    off = n_optcam * 6
    for s in range(3):
        A[2 * i,     off + pt_i * 3 + s] = 1
        A[2 * i + 1, off + pt_i * 3 + s] = 1

    res = least_squares(residual, x0, jac_sparsity=A, method='trf',
                        loss='huber', f_scale=2.0,          # robust to bad matches
                        ftol=1e-4, xtol=1e-4, max_nfev=50, verbose=0)

    ext_final = ext0.copy()
    ext_final[n_fixed:] = res.x[:n_optcam * 6].reshape(-1, 6)
    for k, c in enumerate(cam_ids):
        cameras[c][3:9] = ext_final[k]
    points_3d[:] = res.x[n_optcam * 6:].reshape(-1, 3)
    return cameras, points_3d

def incremental_reconstruction_loop(match_pair, match_inlier, img_keypoints, img_descriptors,
                                   cameras, points_3d, point_visibility, registered_cameras,
                                   best_cam0, best_cam1, f_init, cx_init, cy_init, 
                                   Z_limit, max_cos_parallax, img_set):
    
    failed_cameras = set() # Track cameras that fail with the current point cloud
    iteration = 0
    
    # Stop when all cameras are either registered or proven un-registerable
    while len(registered_cameras) + len(failed_cameras) < len(img_set):
        iteration += 1
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}: Registered cameras: {sorted(registered_cameras)}")
        print(f"Current 3D points: {len(points_3d)}")
        print(f"{'='*70}")
        
        # STEP 1: SELECT NEXT UNREGISTERED CAMERA
        print(f"\nStep 1: Selecting next camera...")
        
        # Pass the failed_cameras set here
        next_cam = find_best_unregistered_camera(match_pair, match_inlier, 
                                                registered_cameras, failed_cameras, img_set)
        
        if next_cam == -1:
            print("✗ No more valid cameras to register")
            break

        print(f"✓ Selected camera {next_cam}")

        # STEP 2: FIND 3D-2D CORRESPONDENCES
        print(f"Step 2: Finding 3D-2D correspondences...")
        object_pts, image_pts, point_3d_indices, next_kp_indices, confidence_scores = \
            get_3d_2d_correspondences_robust(next_cam, registered_cameras, match_pair, 
                                     match_inlier, img_keypoints, img_descriptors, points_3d, 
                                     point_visibility)

        if len(object_pts) < 4:
            print(f"✗ Not enough correspondences ({len(object_pts)} < 4)")
            failed_cameras.add(next_cam) # Add to failed, NOT registered
            continue

        print(f"✓ Found {len(object_pts)} correspondences")
        
        # STEP 3: ESTIMATE POSE WITH PNP
        print(f"Step 3: Estimating camera pose (PnP)...")
        success, inliers, R, t = \
            estimate_pose_with_PnP(next_cam, object_pts, image_pts, cameras, 
                                  f_init, cx_init, cy_init)
        if not success:
            print(f"✗ PnP failed")
            failed_cameras.add(next_cam) # Add to failed, NOT registered
            continue

        cameras[next_cam] = update_camera_pose(cameras[next_cam], R, t)
        print(f"✓ PnP succeeded ({len(inliers)} / {len(object_pts)} inliers)")

        # STEP 4: TRIANGULATE NEW 3D POINTS
        print(f"Step 4: Triangulating new 3D points...")
        new_points_3d, new_point_visibility = \
            triangulate_new_3d_points(next_cam, registered_cameras, match_pair, 
                                     match_inlier, img_keypoints, cameras, 
                                     points_3d, point_visibility, Z_limit, 
                                     max_cos_parallax)

        if len(new_points_3d) > 0:
            points_3d = np.vstack([points_3d, new_points_3d])
            point_visibility = np.vstack([point_visibility, new_point_visibility])
            print(f"✓ Triangulated {len(new_points_3d)} new points")
        else:
            print(f"⊘ No new points triangulated")

        # STEP 5: REFINE CAMERA POSE (LOCAL BA)
        print(f"Step 5: Refining camera pose...")
        cameras[next_cam] = refine_camera_pose(next_cam, cameras, object_pts, 
                                              image_pts, inliers, f_init, cx_init, cy_init)
        print(f"✓ Camera pose refined")

        # STEP 6: REGISTER CAMERA & CLEAR FAILED QUEUE
        registered_cameras.add(next_cam)
        
        # Because we successfully added a camera (and potentially new points), 
        # previously failed cameras might now work! So we clear the failed queue.
        failed_cameras.clear() 
        
        # Update point visibility
        for i, pt_idx in enumerate(point_3d_indices):
            if pt_idx < len(point_visibility):
                point_visibility[pt_idx, next_cam] = next_kp_indices[i]

        print(f"✓ Camera {next_cam} registered")
        print(f"✓ Total 3D points: {len(points_3d)}")

        # STEP 7: GLOBAL BUNDLE ADJUSTMENT (all cameras + all points)
        print(f"Step 7: Global bundle adjustment...")
        cameras, points_3d = run_global_bundle_adjustment(
            cameras, points_3d, point_visibility, registered_cameras, img_keypoints)
        print(f"✓ Global BA done")

    return cameras, points_3d, point_visibility, registered_cameras

def get_point_colors(points_3d, point_visibility, img_keypoints, img_set, registered_cameras):
    """Sample an RGB color for each 3D point from the image it is visible in."""
    colors = np.zeros((len(points_3d), 3), dtype=np.uint8)
    cam_ids = sorted(registered_cameras)
    for pt_idx in range(len(points_3d)):
        for cam_id in cam_ids:
            kp = point_visibility[pt_idx, cam_id]
            if kp != -1:
                x, y = img_keypoints[cam_id][kp].pt
                h, w = img_set[cam_id].shape[:2]
                xi = min(max(int(round(x)), 0), w - 1)
                yi = min(max(int(round(y)), 0), h - 1)
                b, g, r = img_set[cam_id][yi, xi]
                colors[pt_idx] = (r, g, b)   # OpenCV is BGR -> store RGB
                break
    return colors

def save_point_cloud_ply(filename, points_3d, colors):
    """Write a colored point cloud in ASCII PLY format."""
    with open(filename, "wt") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points_3d, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

def visualize_3d_reconstruction(points_3d, cameras, registered_cameras, camera_size=0.5, point_colors=None):
    """
    Visualize the 3D point cloud and camera poses.
    
    Args:
        points_3d: Array of 3D points (N x 3)
        cameras: Array of camera parameters (N_cameras x 9)
        registered_cameras: Set of registered camera indices
        camera_size: Size of camera pyramids for visualization
    """
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    if len(points_3d) > 0:
        if point_colors is not None:
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                      c=point_colors / 255.0, marker='.', s=1, alpha=0.6, label='3D Points')
        else:
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                      c='blue', marker='.', s=1, alpha=0.6, label='3D Points')
    
    # Plot camera poses
    colors = plt.cm.rainbow(np.linspace(0, 1, len(registered_cameras)))
    
    for color_idx, cam_idx in enumerate(sorted(registered_cameras)):
        cam = cameras[cam_idx]
        
        # Extract camera position
        R = Rotation.from_rotvec(cam[3:6]).as_matrix()
        t = cam[6:9]
        cam_pos = -R.T @ t  # Camera center in world coordinates
        
        # Plot camera center
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], 
                  c=[colors[color_idx]], s=100, marker='o', edgecolors='black', linewidths=2)
        
        # Draw camera frame axes (X=red, Y=green, Z=blue)
        axis_length = camera_size
        
        # X-axis (red)
        x_axis = R.T @ np.array([axis_length, 0, 0])
        ax.plot([cam_pos[0], cam_pos[0] + x_axis[0]], 
               [cam_pos[1], cam_pos[1] + x_axis[1]], 
               [cam_pos[2], cam_pos[2] + x_axis[2]], 'r-', linewidth=2, alpha=0.8)
        
        # Y-axis (green)
        y_axis = R.T @ np.array([0, axis_length, 0])
        ax.plot([cam_pos[0], cam_pos[0] + y_axis[0]], 
               [cam_pos[1], cam_pos[1] + y_axis[1]], 
               [cam_pos[2], cam_pos[2] + y_axis[2]], 'g-', linewidth=2, alpha=0.8)
        
        # Z-axis (blue)
        z_axis = R.T @ np.array([0, 0, axis_length])
        ax.plot([cam_pos[0], cam_pos[0] + z_axis[0]], 
               [cam_pos[1], cam_pos[1] + z_axis[1]], 
               [cam_pos[2], cam_pos[2] + z_axis[2]], 'b-', linewidth=2, alpha=0.8)
        
        # Draw camera pyramid (frustum outline)
        f = cam[0]  # Focal length
        pyramid_scale = camera_size / 2
        
        # Near plane corners (in camera frame)
        near_corners = np.array([
            [-pyramid_scale, -pyramid_scale, pyramid_scale],
            [pyramid_scale, -pyramid_scale, pyramid_scale],
            [pyramid_scale, pyramid_scale, pyramid_scale],
            [-pyramid_scale, pyramid_scale, pyramid_scale]
        ])
        
        # Transform to world frame
        near_corners_world = R.T @ near_corners.T + cam_pos[:, np.newaxis]
        
        # Draw pyramid edges
        for i in range(4):
            # Draw near plane edges
            next_i = (i + 1) % 4
            ax.plot([near_corners_world[0, i], near_corners_world[0, next_i]],
                   [near_corners_world[1, i], near_corners_world[1, next_i]],
                   [near_corners_world[2, i], near_corners_world[2, next_i]],
                   color=colors[color_idx], linewidth=1, alpha=0.7)
            
            # Draw lines from camera center to near plane
            ax.plot([cam_pos[0], near_corners_world[0, i]],
                   [cam_pos[1], near_corners_world[1, i]],
                   [cam_pos[2], near_corners_world[2, i]],
                   color=colors[color_idx], linewidth=0.5, alpha=0.5)
        
        # Label camera
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f'  C{cam_idx}', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'3D Reconstruction: {len(points_3d)} points, {len(registered_cameras)} cameras')
    
    # Set equal aspect ratio
    max_range = np.array([points_3d[:, 0].max()-points_3d[:, 0].min(),
                         points_3d[:, 1].max()-points_3d[:, 1].min(),
                         points_3d[:, 2].max()-points_3d[:, 2].min()]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

def main():
    img_path = "data/relief/%02d.jpg"
    img_resize = 0.25
    f_init, cx_init, cy_init, Z_init, Z_limit = 500, -1, -1, 2, 100
    max_cos_parallax = np.cos(10*np.pi / 180)
    min_inlier_num = 200
    SHOW_MATCH = False

    print("="*70)
    print("INCREMENTAL STRUCTURE-FROM-MOTION")
    print("="*70)

    # ========================================================================
    # STAGE 1: LOAD IMAGES & EXTRACT FEATURES
    # ========================================================================
    print("\n[Stage 1] Loading images and extracting features...")
    
    img_keypoints = []
    img_descriptors = []
    img_set = []
    detector = cv2.BRISK_create()
    cam = cv2.VideoCapture(img_path)

    while True:
        _, img = cam.read()
        if img is None:
            break
        img = cv2.resize(img, dsize=(0, 0), fx=img_resize, fy=img_resize)
        img_keypoint, img_descriptor = detector.detectAndCompute(img, None)
        img_keypoints.append(img_keypoint)
        img_descriptors.append(img_descriptor)
        img_set.append(img)
    cam.release()

    if cx_init < 0:
        cx_init = int(img_set[0].shape[1] / 2)
    if cy_init < 0:
        cy_init = int(img_set[0].shape[0] / 2)

    img_keypoints = np.array(img_keypoints, dtype=object)
    img_descriptors = np.array(img_descriptors, dtype=object)
    img_set = np.array(img_set)
    
    print(f"✓ Loaded {len(img_set)} images")

    # ========================================================================
    # STAGE 2: FEATURE MATCHING
    # ========================================================================
    print("\n[Stage 2] Feature matching...")
    
    fmatcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    match_pair, match_inlier = [], []
    
    for i in range(len(img_set)):
        for j in range(i+1, len(img_set)):
            src, dst, inlier = [], [], []
            match = fmatcher.match(img_descriptors[i], img_descriptors[j])
            
            if len(match) < 8:
                continue
            
            match = np.array(match)
            for m in match:
                src.append(img_keypoints[i][m.queryIdx].pt)
                dst.append(img_keypoints[j][m.trainIdx].pt)

            src = np.array(src, dtype=np.float32)
            dst = np.array(dst, dtype=np.float32)

            F, inlier_mask = cv2.findFundamentalMat(src, dst, cv2.RANSAC)
            
            if inlier_mask is None:
                continue

            inlier_mask = inlier_mask.flatten()
            for k in range(len(inlier_mask)):
                if inlier_mask[k]:
                    inlier.append(match[k])

            inlier = np.array(inlier)
            
            if inlier.size < min_inlier_num:
                continue

            print(f"  Image {i} - {j}: {inlier.size} inliers")
            match_pair.append((i, j))
            match_inlier.append(inlier)

    if len(match_pair) < 1:
        print("✗ No good image pairs found")
        return 
    
    print(f"✓ Found {len(match_pair)} image pairs")

    # ========================================================================
    # STAGE 3: FIND BEST INITIAL PAIR
    # ========================================================================
    print("\n[Stage 3] Finding best initial pair...")
    
    cameras = np.full((len(img_set), 9), 
                     np.array([f_init, cx_init, cy_init, 0, 0, 0, 0, 0, 0]), 
                     dtype=np.float32)

    best_pair_idx = -1
    best_3d_count = 0
    best_points_3d = None
    best_cam0_state = None
    best_cam1_state = None
    best_kp0_indices = []
    best_kp1_indices = []

    for pair_idx, (cam0_idx, cam1_idx) in enumerate(match_pair):
        src, dst = [], []
        for match in match_inlier[pair_idx]:
            src.append(img_keypoints[cam0_idx][match.queryIdx].pt)
            dst.append(img_keypoints[cam1_idx][match.trainIdx].pt)
        
        src = np.array(src, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)

        E, inlier_mask = cv2.findEssentialMat(src, dst, f_init, (cx_init, cy_init), 
                                             cv2.RANSAC, 0.999, 1.0)
        if E is None:
            continue
        
        inlier_mask = inlier_mask.flatten()
        _, R, t, _ = cv2.recoverPose(E, src, dst, mask=inlier_mask.reshape(-1, 1))

        # Filter outliers
        clean_src, clean_dst, clean_kp0, clean_kp1 = [], [], [], []
        for r in range(len(inlier_mask)):
            if inlier_mask[r]:
                clean_src.append(src[r])
                clean_dst.append(dst[r])
                # Track original keypoints to populate visibility matrix
                clean_kp0.append(match_inlier[pair_idx][r].queryIdx)
                clean_kp1.append(match_inlier[pair_idx][r].trainIdx)

        clean_src = np.array(clean_src, dtype=np.float32)
        clean_dst = np.array(clean_dst, dtype=np.float32) # Fixed typo: was dst

        temp_cam0 = np.array([f_init, cx_init, cy_init, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        temp_cam1 = np.array([f_init, cx_init, cy_init, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        temp_cam1 = update_camera_pose(temp_cam1, R, t)

        P0 = get_projection_mat(temp_cam0)
        P1 = get_projection_mat(temp_cam1)
        pts_4d = cv2.triangulatePoints(P0, P1, clean_src.T, clean_dst.T)
        pts_4d = pts_4d / pts_4d[3, :]
        pts_3d = pts_4d[:3, :].T

        # Filter out bad points AND track the keypoint indices that survive
        valid_pts_3d = []
        valid_kp0 = []
        valid_kp1 = []
        
        for p_idx, p in enumerate(pts_3d):
            if not isBadPoint(p, temp_cam0, temp_cam1, Z_limit, max_cos_parallax):
                valid_pts_3d.append(p)
                valid_kp0.append(clean_kp0[p_idx])
                valid_kp1.append(clean_kp1[p_idx])

        good_count = len(valid_pts_3d)
        print(f"  Pair {cam0_idx}-{cam1_idx}: {good_count} valid 3D points")

        if good_count > best_3d_count:
            best_3d_count = good_count
            best_points_3d = np.array(valid_pts_3d, dtype=np.float32)
            best_cam0_state = temp_cam0.copy()
            best_cam1_state = temp_cam1.copy()
            best_pair_idx = pair_idx
            best_kp0_indices = valid_kp0.copy() # Save the winning keypoint indices
            best_kp1_indices = valid_kp1.copy()

    if best_3d_count < 100:
        print(f"✗ Best pair only has {best_3d_count} valid 3D points (need >= 100)")
        return

    best_cam0, best_cam1 = match_pair[best_pair_idx]
    cameras[best_cam0] = best_cam0_state
    cameras[best_cam1] = best_cam1_state
    
    # Initialize points_3d and point_visibility
    points_3d = best_points_3d.copy()
    point_visibility = np.full((len(points_3d), len(img_set)), -1, dtype=int)

    for idx in range(len(points_3d)):
        # Assign the actual keypoint indices
        point_visibility[idx, best_cam0] = best_kp0_indices[idx]
        point_visibility[idx, best_cam1] = best_kp1_indices[idx]    
    
    print(f"✓ Selected pair: Image {best_cam0} - {best_cam1}")
    print(f"✓ Initialized with {best_3d_count} 3D points")

    # ========================================================================
    # STAGE 4: INCREMENTAL REGISTRATION
    # ========================================================================
    print("\n[Stage 4] Starting incremental registration...")
    
    points_3d = best_points_3d.copy()
    registered_cameras = {best_cam0, best_cam1}
    # point_visibility = np.zeros((len(points_3d), len(img_set)), dtype=np.uint8)

    for idx in range(len(points_3d)):
        point_visibility[idx, best_cam0] = best_kp0_indices[idx]
        point_visibility[idx, best_cam1] = best_kp1_indices[idx]

    # Run incremental loop
    cameras, points_3d, point_visibility, registered_cameras = incremental_reconstruction_loop(
        match_pair, match_inlier, img_keypoints, img_descriptors, 
        cameras, points_3d, point_visibility, registered_cameras,
        best_cam0, best_cam1, f_init, cx_init, cy_init, 
        Z_limit, max_cos_parallax, img_set
    )

    # ========================================================================
    # STAGE 5: SAVE RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("✓ RECONSTRUCTION COMPLETED")
    print("="*70)
    print(f"  Final 3D points: {len(points_3d)}")
    print(f"  Registered cameras: {len(registered_cameras)}/{len(img_set)}")

    print("\n[Saving results...]")
    
    # Sample a color per 3D point from the source images
    point_colors = get_point_colors(points_3d, point_visibility, img_keypoints,
                                     img_set, registered_cameras)

    # Save 3D points (x y z r g b)
    points_3d_file = "sfm_incremental_points.xyz"
    with open(points_3d_file, "wt") as f:
        for point, c in zip(points_3d, point_colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
    print(f"✓ Saved 3D points to {points_3d_file}")

    # Save colored point cloud in PLY format
    points_ply_file = "sfm_incremental_points.ply"
    save_point_cloud_ply(points_ply_file, points_3d, point_colors)
    print(f"✓ Saved colored point cloud to {points_ply_file}")

    # Save camera poses
    camera_file = "sfm_incremental_cameras.xyz"
    with open(camera_file, 'wt') as f:
        for cam_idx in sorted(registered_cameras):
            cam = cameras[cam_idx]
            f.write(f"{cam[0]:.6f} {cam[1]:.6f} {cam[2]:.6f} ")
            f.write(f"{cam[3]:.6f} {cam[4]:.6f} {cam[5]:.6f} ")
            f.write(f"{cam[6]:.6f} {cam[7]:.6f} {cam[8]:.6f}\n")
    print(f"✓ Saved camera poses to {camera_file}")
    
    visualize_3d_reconstruction(points_3d, cameras, registered_cameras, camera_size=0.3,
                                point_colors=point_colors)


if __name__ == "__main__":
    main()