import numpy as np
import os
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def pack_params(K, R, t):
    """
    Pack parameters for camera calibration.
    """
    if (K[0, 0] != K[1, 1]):
        raise ValueError("Currently only supports fx = fy. Please ensure K[0, 0] == K[1, 1].")
    
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    rvec = Rotation.from_matrix(R).as_rotvec()
    tvec = t.flatten()
    return np.concatenate(([f, cx, cy], rvec, tvec))

def unpack_params(params):
    """
    Unpack parameters for camera calibration.
    """
    k_matrix_offset = 3
    f, cx, cy = params[0:k_matrix_offset]
    rvec = params[k_matrix_offset:7]
    tvec = params[k_matrix_offset+3:10]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    R = Rotation.from_rotvec(rvec).as_matrix()
    return K, R, tvec

def pack_params_dist(K, dist_coeffs , R, t):
    """
    Pack parameters for camera calibration including distortion coefficients.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    rvec = Rotation.from_matrix(R).as_rotvec()
    tvec = t.flatten()
    return np.concatenate(([fx, fy, cx, cy], dist_coeffs, rvec, tvec))

def unpack_params_dist(params):
    """
    Unpack parameters including distortion coefficients.
    """
    fx, fy, cx, cy = params[0:4]
    dist_coeffs = params[4:9]
    rvec = params[9:12]
    tvec = params[12:15]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R = Rotation.from_rotvec(rvec).as_matrix()
    return K, R, tvec, dist_coeffs


def load_files_with_fallback(filenames, data_paths=['data', '../data']):
    for data_path in data_paths:
        try:
            file_paths = [os.path.join(data_path, filename) for filename in filenames]
            
            # Check if all files exist before loading
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load all files
            loaded_data = [np.loadtxt(file_path, dtype=np.float32) for file_path in file_paths]
            return loaded_data, data_path
            
        except (FileNotFoundError, OSError, IOError):
            continue  # Try next path
    
    # If we reach here, no path worked
    raise FileNotFoundError(
        f"Could not find files {filenames} in any of these paths: {data_paths}\n"
        f"Tried: {', '.join([os.path.join(path, filenames[0]) for path in data_paths])}"
    )