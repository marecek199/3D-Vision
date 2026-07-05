import numpy as np
import cv2 as cv

import sys; import os;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cv_engine.detector_dnn import DetectorDnn

# Instantiate feature detector and matcher
video_file, obj_file = 'data/blais.mp4', 'data/blais.jpg'
f, cx, cy = 1000, 320, 240
min_inlier_num = 4

detector = DetectorDnn(
    detector='SUPERPOINT',
    confidence_threshold=0.1,
    processing_size=-1,
    max_keypoints=2048
    )

# Load the object image and extract features
obj_image = cv.imread(obj_file)

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given video, ' + video_file

# Prepare a box for simple AR
box_lower = np.array([[30, 145, 0], [30, 200, 0], [200, 200, 0], [200, 145, 0]], dtype=np.float32)
box_upper = np.array([[30, 145, -50], [30, 200, -50], [200, 200, -50], [200, 145, -50]], dtype=np.float32)

# Run pose extimation
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
dist_coeff = np.zeros(5)
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Detect and match features between the object and the image
    # Extract features and match them to the object features
    img_keypoints, obj_keypoints, matches = detector.run(img, obj_image)

    if len(matches) < min_inlier_num:
        print("Not enough matches to compute pose.")
        continue

    obj_pts, img_pts = [], []
    for m in matches:
        img_pts.append(img_keypoints[m[0]].astype(int))
        obj_pts.append(obj_keypoints[m[1]].astype(int))
    obj_pts = np.array(obj_pts, dtype=np.float32)
    obj_pts = np.hstack((obj_pts, np.zeros((len(obj_pts), 1), dtype=np.float32))) # Make 2D to 3D
    img_pts = np.array(img_pts, dtype=np.float32)

    if len(obj_pts) < 4 or len(img_pts) < 4:
        print("Not enough points for solvePnPRansac.")
        continue

    # Deterimine whether each matched feature is an inlier or not
    ret, rvec, tvec, inliers = cv.solvePnPRansac(obj_pts, img_pts, K, dist_coeff, useExtrinsicGuess=False,
                                                 iterationsCount=500, reprojectionError=2., confidence=0.99)
    inlier_mask = np.zeros(len(matches), dtype=np.uint8)
    inlier_mask[inliers] = 1
    
    # Convert numpy arrays to cv.KeyPoint objects
    img_keypoints_cv = [cv.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in img_keypoints]
    obj_keypoints_cv = [cv.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in obj_keypoints]
    cv_matches = [cv.DMatch(_queryIdx=int(m[0]), _trainIdx=int(m[1]), _distance=0) for m in matches]

    img_result = cv.drawMatches(img, img_keypoints_cv, obj_image, obj_keypoints_cv, cv_matches, None, (0, 0, 255), (0, 127, 0), inlier_mask)

    # Check whether inliers are enough or not
    inlier_num = sum(inlier_mask)
    if inlier_num > min_inlier_num:
        # Estimate camera pose with inliers
        ret, rvec, tvec = cv.solvePnP(obj_pts[inliers.ravel()], img_pts[inliers.ravel()], K, dist_coeff)

        # Draw the box on the image
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        line_lower = np.int32(line_lower)
        line_upper = np.int32(line_upper)
        cv.polylines(img_result, [line_lower], True, (255, 0, 0), 2)
        cv.polylines(img_result, [line_upper], True, (0, 0, 255), 2)
        for b, t in zip(line_lower, line_upper):
            cv.line(img_result, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

    # Show the image and process the key event
    info = f'Inliers: {inlier_num} ({inlier_num*100/len(matches):.0f}%), Focal length: {f}'
    cv.putText(img_result, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    cv.imshow('Pose Estimation (Book)', img_result)
    key = cv.waitKey(1)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()
