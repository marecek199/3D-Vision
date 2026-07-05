import numpy as np
import cv2 as cv
import time


class Detector:
       
    def __init__(self, detector_type='ORB', number_of_features=1000):
                
        self._name = str.upper(detector_type)
        self.number_of_features = number_of_features
        self.info = str()
        self._keypoints1, self._keypoints2, self._matches = [], [], []
        
        if detector_type == 'ORB':
            self.detector = cv.ORB_create(number_of_features)
            self.matcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
        elif detector_type == 'SIFT':
            self.detector = cv.SIFT_create(number_of_features)
            self.matcher = cv.DescriptorMatcher_create('BruteForce')
        elif detector_type == 'AKAZE':
            self.detector = cv.AKAZE_create(number_of_features)
            self.matcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
        elif detector_type == 'BRISK':
            self.detector = cv.BRISK_create(thresh=30, octaves=3, patternScale=1.0)
            self.matcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
        elif detector_type == 'FAST':
            self.detector = cv.FastFeatureDetector_create(number_of_features)
            self.matcher = None
        elif detector_type == 'GFTT': # Good features to track
            self.detector = cv.GFTTDetector_create(number_of_features)
            self.matcher = None
        elif detector_type == 'KAZE':
            self.detector = cv.KAZE_create(number_of_features)
            self.matcher = None
        elif detector_type == 'KLT':
            self.detector = None
            self.matcher = None
        elif detector_type == 'MSER':
            self.detector = cv.MSER_create(number_of_features)
            self.matcher = None
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
        

    def run(self, image1, image2, match_ratio_threshold=-1, max_matches=0):
        self.reset()
        
        time_start = time.time()
        
        # Detect feature points
        self._keypoints1, descriptors1 = self.extract_features(image1)
        self._keypoints2, descriptors2 = self.extract_features(image2)
        time_detect = time.time()
        
        self._matches = self.match_features(descriptors1, descriptors2, match_ratio_threshold, max_matches)
        
        time_match = time.time()
        
        self.info = self._name + f': {len(self._matches)} matches ({(time_detect-time_start)*1000:.0f} + {(time_match-time_detect)*1000:.0f} = {(time_match-time_start)*1000:.0f} [msec])'
        
        return self._keypoints1, self._keypoints2, self._matches


    def get_matched_points(self):                
        if (not self._keypoints1) or (not self._keypoints2) or (not self._matches):
            raise ValueError("Keypoints or matches are not computed. Run the detector first.")        
        
        pts1 = np.float32([self._keypoints1[m.queryIdx].pt for m in self._matches]).reshape(-1, 2)
        pts2 = np.float32([self._keypoints2[m.trainIdx].pt for m in self._matches]).reshape(-1, 2)
        return pts1, pts2

    def compute_keypoints(self, image):
        if self.detector is None:
            raise ValueError("Detector is not defined for this detector type.")
        keypoints = self.detector.detect(image)
        return keypoints

    def extract_features(self, image):
        keypoints = self.compute_keypoints(image)
        keypoints, descriptors = self.detector.compute(image, keypoints)
        return keypoints, descriptors

    def match_features(self, descriptors1, descriptors2, match_ratio_threshold=-1, max_matches=0):
        # Match features
        if self.matcher is None:
            raise ValueError("Matcher is not defined for this detector type.")

        # Validate descriptors
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
            return []

        # ===== BASIC MATCHING (no filtering) =====
        if match_ratio_threshold <= 0:
            print(f"Using basic matching (no Lowe's ratio test)")
            matches = self.matcher.match(descriptors1, descriptors2)
            
        # ===== LOWE'S RATIO TEST (with filtering) =====
        elif match_ratio_threshold > 0:
            # Use knnMatch for Lowe's ratio test
            matches_knn = self.matcher.knnMatch(descriptors1, descriptors2, k=2)        
            
            # Apply Lowe's ratio test
            matches = []
            for match_pair in matches_knn:
                if len(match_pair) == 2:
                    m, n = match_pair
                    # m = best match, n = 2nd best match
                    # Only keep if best is significantly better than 2nd best
                    if m.distance < match_ratio_threshold * n.distance:
                        matches.append(m)
                elif len(match_pair) == 1:
                    # Only one match found (no 2nd best to compare)
                    # Accept it if threshold allows
                    matches.append(match_pair[0])
        else:
            raise ValueError(f"Invalid match_ratio_threshold: {match_ratio_threshold}. Must be -1 or >= 0")
            
        # Sort by distance and optionally limit
        self._matches = sorted(matches, key=lambda x: x.distance)
        
        # Limit to max_matches if specified
        if max_matches is not None and max_matches > 0:
            self._matches = self._matches[:max_matches]
            
        return self._matches

    def reset(self):
        self._keypoints1, self._keypoints2, self._matches = [], [], []
        self.info = str()
        
    def hasMatcher(self):
        return self.matcher is not None