import numpy as np
from scipy.spatial.transform import Rotation
from . import geometry 
from . import utilsEngine

def reprojection_error(params, X, x):
    K, R, tvec = utilsEngine.unpack_params(params)
    projected_points = geometry.project(X, K, R, tvec)
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
        projected_points = geometry.project(X, K, R, tvec)
        
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

def cost_function(params, obj_pts, img_pts):
    # Compute the cost (sum of squared reprojection errors)    
    return np.sum(reprojection_error(params, obj_pts, img_pts)**2)

def cost_function_multiple_views(params, obj_pts, img_pts):
    # Compute the cost (sum of squared reprojection_error_multiple_views errors)    
    return np.sum(reprojection_error_multiple_views(params, obj_pts, img_pts)**2)

def my_approx_fprime(params, func, epsilon, *args):
    grad = np.zeros_like(params)
    
    for i in range(len(params)):
        params_plus_epsilon = params.copy()
        params_minus_epsilon = params.copy()
        
        params_plus_epsilon[i] += epsilon
        params_minus_epsilon[i] -= epsilon
        
        f_x_plus_epsilon = func(params_plus_epsilon, *args)
        f_x_minus_epsilon = func(params_minus_epsilon, *args)
        
        grad[i] = (f_x_plus_epsilon - f_x_minus_epsilon) / (2 * epsilon)
                
    return grad

def gradient_optimizer( params, obj_pts, img_pts, cost_fun = cost_function, num_iterations=30000):
    learning_rate = 1e-3
    tolerance = 1e-8
    prev_cost = float('inf')
        
    for i in range(num_iterations):
        cost = cost_fun(params, obj_pts, img_pts)
        
        #  Adjust learning rate dynamically
        if i > (0.5  * num_iterations):
            learning_rate = 1e-4  # Decrease learning rate after half iterations
        elif i > (0.8  * num_iterations):
            learning_rate = 1e-5  # Further decrease learning rate
        
        if i % 100 == 0:
            print(f"Iteration {i:3d} | Cost: {cost:.4f}")

        # Check for convergence
        if np.all(abs(cost - prev_cost) < tolerance):
            print(f"Converged at iteration {i:3d} with cost: {cost:.4f}")
            break
        prev_cost = cost

        # Compute gradient using finite differences
        grad = my_approx_fprime(params, cost_fun, 1e-4, obj_pts, img_pts)

        # Gradient clipping to prevent exploding gradients
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 10.0:
            grad = grad / grad_norm

        # Update parameters
        params = params - learning_rate * grad
        
        # Ensure K parameters stay positive (f, cx, cy should be > 0)
        params[0] = max(params[0], 1.0)  # f
        params[1] = max(params[1], 1.0)  # cx
        params[2] = max(params[2], 1.0)  # cy
        
    print(f"Iteration {num_iterations:3d} | Final error: {cost_fun(params, obj_pts, img_pts):.4f}")
    return params