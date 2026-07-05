import numpy as np
import cv2 as cv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_engine.geometry as geometry

def warpPerspective(src, H, dst_size):
    '''
    Forward Mapping: Source -> Destination
    Note: This implementation may create holes in the output image.
    '''
    # Create an output image
    width, height = dst_size
    channels = src.shape[2] if src.ndim > 2 else 1
    dst = np.zeros((height, width, channels), dtype=src.dtype)
    
    # Map each pixel from source to destination
    for py in range(src.shape[0]):
        for px in range(src.shape[1]):
            # Project source pixel (px, py) to destination (qx, qy)
            q = H @ [px, py, 1]
            # Normalize homogeneous coordinates
            qx, qy = int(q[0]/q[-1] + 0.5), int(q[1]/q[-1] + 0.5)
            # Check bounds and assign pixel value
            if 0 <= qx < width and 0 <= qy < height:
                dst[qy, qx] = src[py, px]
                                
    return dst
            

if __name__ == '__main__':
    try:
        img = cv.imread('data/sunglok_card.jpg')
    except:
        img = cv.imread('../data/sunglok_card.jpg')
    
    wnd_name = 'Image Warping'
    card_size = (900, 480)
    pts_src = np.array([[95, 243], [743, 121], [157, 652], [969, 372]], dtype=np.float32)
    pts_dst = np.array([[0, 0], [card_size[0], 0], [0, card_size[1]], card_size], dtype=np.float32)

    img_copy = img.copy()
    for px,py in pts_src:
        cv.circle(img_copy, (int(px), int(py)), 3, color=(0, 0, 255), thickness=3)

    # Find planar homography and transform the original image
    H = geometry.calibrate_DLT_homography(pts_src, pts_dst)
    warp1 = warpPerspective(img, H, card_size)
    warp = geometry.warpPerspective(img, H, card_size)

    # Show images generated from two methods
    cv.imshow('Source Image with Points', img_copy)
    cv.imshow(wnd_name + ' (Method 1)', warp1)
    cv.imshow(wnd_name + ' (Method 2)', warp)
    cv.waitKey(0)
    cv.destroyAllWindows()