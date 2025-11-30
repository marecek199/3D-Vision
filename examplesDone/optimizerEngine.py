import numpy as np
import scipy.linalg
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import scipy
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

import utilsEngine as ue

# --------------------------------------------------------------------
# Optimization functions 
# --------------------------------------------------------------------

def cost_function(params, obj_pts, img_pts):
    # Compute the cost (sum of squared reprojection errors)    
    return np.sum(ue.reprojection_error(params, obj_pts, img_pts)**2)

def cost_function_multiple_views(params, obj_pts, img_pts):
    # Compute the cost (sum of squared reprojection_error_multiple_views errors)    
    return np.sum(ue.reprojection_error_multiple_views(params, obj_pts, img_pts)**2)

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