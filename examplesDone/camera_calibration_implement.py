import numpy as np
from scipy.optimize import least_squares
import computerVisionEngine as cve

if __name__ == '__main__':
    img_size = (640, 480)
    try:
        img_files = ['data/image_formation1.xyz', 'data/image_formation2.xyz']
        img_pts = []
        for file in img_files:
            pts = np.loadtxt(file, dtype=np.float32)
            img_pts.append(pts[:,:2])
        pts = np.loadtxt('data/box.xyz', dtype=np.float32)
    except Exception as e:
        img_files = ['../data/image_formation1.xyz', '../data/image_formation2.xyz']
        img_pts = []
        for file in img_files:
            pts = np.loadtxt(file, dtype=np.float32)
            img_pts.append(pts[:,:2])
        pts = np.loadtxt('../data/box.xyz', dtype=np.float32)

    obj_pts = [pts] * len(img_pts) # Copy the object point as much as the number of image observation

    # Calibrate the camera
    _, K, *_ = cve.calibrateCamera(obj_pts, img_pts, img_size)

    print('\n### Ground Truth')
    print('* f, cx, cy = 1000, 320, 240')
    print('\n### My Calibration')
    print(f'* f, cx, cy = {K[0,0]:.1f}, {K[0,2]:.1f}, {K[1,2]:.1f}')
