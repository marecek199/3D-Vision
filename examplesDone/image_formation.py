import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry as geometry

# The default camera configuration: Focal length, principal point, image resolution, position, and orientation
f, cx, cy, noise_std = 1000, 320, 240, 1
img_res = (640, 480)
cam_pos = [[0, 0, 0], [-2, -2, 0], [2, 2, 0], [-2, 2, 0], [2, -2, 0]]          # Unit: [m]
cam_ori = [[0, 0, 0], [-15 , 15, 0], [15, -15, 0], [15, 15, 0], [-15, -15, 0]] # Unit: [deg]

# Load a point cloud in the homogeneous coordinate
try:
    X = np.loadtxt('../data/box.xyz') # N x 3
except Exception as e:
    X = np.loadtxt('data/box.xyz') # N x 3

# Generate images for each camera pose
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
for i, (pos, ori) in enumerate(zip(cam_pos, cam_ori)):
    # Derive 'R' and 't'
    R_world = Rotation.from_euler('xyz', ori, degrees=True).as_matrix()
    t_world = np.array(pos).reshape(-1, 1)
    
    R_cam = R_world.T
    t_cam = -R_world.T @ t_world
    
    X_cam = K @ (R_cam @ X.T + t_cam)  # 3 x N
    X_cam /= X_cam[-1, :] # Normalize and dehomogenize
    
    # Create and save the image
    img = np.zeros((img_res[1], img_res[0], 3), dtype=np.uint8)  # 3-channel image
    for x in X_cam[:2, :].T:
        cv.circle(img, x.astype(np.int32), 2, color=(0, 0, 255), thickness=3)
        
    cv.imshow(f'Image Formation {i}', img)
    np.savetxt(f'image_formation{i}.xyz', X_cam.T) # N x 2
    
cv.waitKey()
cv.destroyAllWindows()
    
