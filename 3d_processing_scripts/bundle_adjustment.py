import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
import argparse

f, cx, cy = 1000, 320, 240

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """
    Create sparsity matrix for Jacobian.
    
    Args:
        n_cameras: Number of camera views
        n_points: Number of 3D points
        camera_indices: Array mapping each observation to camera index
        point_indices: Array mapping each observation to 3D point index
    
    Returns:
        Sparse matrix A (m x n) where:
        - m = 2 * num_observations (2D points)
        - n = 6*n_cameras + 3*n_points (parameters)
    """
    
    m = camera_indices.size * 2  # Each observation has 2 equations (u, v)
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    
    # Each observation depends on 6 camera parameters    
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    # Each observation depends on 3 point parameters
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

def project(points_3d, camera_params):
    """
    Project 3D points using camera parameters.
    
    Args:
        points_3d: Shape (N, 3) - 3D points in world coordinates
        camera_params: Shape (N, 6) - [rvec (3), tvec (3)] per camera
                      rvec is rotation vector (axis-angle)
    
    Returns:
        projected_2d: Shape (N*2,) - flattened [u1, v1, u2, v2, ...]
    """
    
    global f, cx, cy
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

    # Extract rotation vectors and translation vectors
    rot_vecs = camera_params[:, :3]  # Shape: (N, 3)
    t_vecs = camera_params[:, 3:]    # Shape: (N, 3)

    # Convert rotation vector to rotation matrix using Rodrigues formula
    # θ = ||rot_vec||, v = rot_vec / θ
    # R = cos(θ)I + sin(θ)[v]_x + (1-cos(θ))vv^T
    theta = np.linalg.norm(rot_vecs, axis=1, keepdims=True)
    
    # Handle zero rotation (avoid division by zero)
    with np.errstate(invalid='ignore', divide='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    
    # Rodrigues formula components
    dot = np.sum(points_3d * v, axis=1, keepdims=True)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Apply rotation: R @ X
    points_rotated = (cos_theta * points_3d + 
                      sin_theta * np.cross(v, points_3d) + 
                      dot * (1 - cos_theta) * v)

    # Apply translation: R @ X + t
    points_cam = points_rotated + t_vecs
    
    # Project to image: K @ X_cam
    points_proj = points_cam @ K.T
    
    # Normalize by z-coordinate (perspective division)
    points_proj = points_proj / points_proj[:, 2:3]
    
    # Return only x, y coordinates (u, v), flattened
    return points_proj[:, :2].ravel()

def residual_function(params, points_2d_observed, n_cameras, n_points, 
                     camera_indices, point_indices):
    """
    Compute residuals (differences) for all observations.
    
    Args:
        params: Flattened parameter vector [camera params, 3D points]
        points_2d_observed: Flattened observed 2D points
        n_cameras, n_points: Dimensions
        camera_indices, point_indices: Index arrays
    
    Returns:
        residuals: Difference between observed and projected points
    """
    
    # Extract camera parameters and 3D points from parameter vector
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    # Get relevant 3D points for each observation
    points_3d_obs = points_3d[point_indices]  # Shape: (num_obs, 3)
    
    # Get relevant camera parameters for each observation
    camera_params_obs = camera_params[camera_indices]  # Shape: (num_obs, 6)
    
    # Project 3D points to 2D
    points_2d_projected = project(points_3d_obs, camera_params_obs)
    
    # Compute residuals: observed - projected
    residuals = points_2d_observed - points_2d_projected
    
    return residuals

def bundle_adjustment(image_points, n_cameras, n_points, camera_indices, 
                     point_indices, use_jacobian_sparsity=True):
    """
    Perform bundle adjustment optimization.
    
    Args:
        image_points: Observed 2D points (num_observations, 2)
        n_cameras: Number of cameras
        n_points: Number of 3D points
        camera_indices: Camera index for each observation
        point_indices: 3D point index for each observation
        use_jacobian_sparsity: Use sparse Jacobian for efficiency
    
    Returns:
        result: Optimization result from least_squares
    """
    
    # Flatten observed points
    points_2d_flat = image_points.ravel()
    
    # Initialize camera parameters (6 params each)
    cameras_init = np.zeros((n_cameras, 6))
    cameras_init[:, 2] = 1  # Look forward (rotation vector pointing along z)
    
    # Initialize 3D points
    points_3d_init = np.full((n_points, 3), [0, 0, 5.5])  # Points at z=5.5
    
    # Combine into parameter vector
    x0 = np.hstack((cameras_init.ravel(), points_3d_init.ravel()))
    
    # Create sparsity pattern if requested
    jac_sparsity = None
    if use_jacobian_sparsity:
        jac_sparsity = bundle_adjustment_sparsity(n_cameras, n_points, 
                                                  camera_indices, point_indices)
    
    # Run least squares optimization
    print("Starting bundle adjustment optimization...")
    t_start = time.time()
    
    result = least_squares(
        residual_function,
        x0,
        args=(points_2d_flat, n_cameras, n_points, camera_indices, point_indices),
        jac_sparsity=jac_sparsity,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=100,
        verbose=1,
        method='trf'  # Trust Region Reflective
    )
    
    t_end = time.time()
    print(f"Optimization completed in {t_end - t_start:.2f} seconds")
    print(f"Final cost: {result.cost:.6f}")
    
    return result


def main():
    """Load data, setup indices, run bundle adjustment, and save results."""
    
    # Load 2D observations from multiple views
    input_num = 5
    xs = []
    for i in range(input_num):
        try:
            filename = f"data/image_formation{i}.xyz"
            data = np.genfromtxt(filename, delimiter=" ")
            xs.append(data[:, :2])  # Keep only x, y (ignore z if present)
        except FileNotFoundError:
            print(f"Warning: Could not find {filename}")
            continue
    
    if not xs:
        print("Error: No image data files found!")
        return
    
    xs = np.array(xs)
    n_cameras = xs.shape[0]
    n_points = xs.shape[1]
    
    print(f"Loaded {n_cameras} camera views with {n_points} points each")
    

    # camera_indices[i] = which camera made observation i
    # point_indices[i] = which 3D point is observation i    
    camera_indices = np.array([], dtype=int)
    point_indices = np.array([], dtype=int)
    
    for cam_idx in range(n_cameras):
        # For this camera, we observe all points
        camera_indices = np.hstack((camera_indices, 
                                   np.full(n_points, cam_idx, dtype=int)))
        point_indices = np.hstack((point_indices, 
                                  np.arange(n_points, dtype=int)))
    
    # Run Bundle Adjustment
    result = bundle_adjustment(
        xs.reshape(-1, 2),  # Flatten to (n_obs, 2)
        n_cameras, 
        n_points,
        camera_indices,
        point_indices,
        use_jacobian_sparsity=True
    )
    
    opt_cameras = result.x[:n_cameras * 6].reshape((n_cameras, 6))
    opt_points = result.x[n_cameras * 6:].reshape((n_points, 3))
    
    point_file = "bundle_adjustment_global(point)_by_myself.xyz"
    with open(point_file, 'w') as f:
        for i in range(n_points):
            f.write(f"{opt_points[i, 0]} {opt_points[i, 1]} {opt_points[i, 2]}\n")
    print(f"Saved 3D points to {point_file}")
    
    camera_file = "bundle_adjustment_global(camera)_by_myself.xyz"
    with open(camera_file, 'w') as f:
        for i in range(n_cameras):
            f.write(f"{opt_cameras[i, 0]} {opt_cameras[i, 1]} {opt_cameras[i, 2]} "
                   f"{opt_cameras[i, 3]} {opt_cameras[i, 4]} {opt_cameras[i, 5]}\n")
    print(f"Saved camera poses to {camera_file}")
    
    # Compute cost before and after optimization
    x0 = np.zeros_like(result.x)
    x0[:n_cameras * 6] = 0
    x0[n_cameras * 6:] = 5.5
    
    cost_before = np.sum(residual_function(x0, xs.reshape(-1, 2).reshape(-1,), n_cameras, 
                                          n_points, camera_indices, 
                                          point_indices) ** 2)
    cost_after = result.cost
    
    print(f"\nOptimization Results:")
    print(f"  Cost before: {cost_before:.6f}")
    print(f"  Cost after:  {cost_after:.6f}")
    print(f"  Improvement: {cost_before - cost_after:.6f}")


if __name__ == "__main__":
    main()