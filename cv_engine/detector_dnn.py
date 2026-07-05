import torch
import cv2
import numpy as np
import time
from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT, DoGHardNet
from lightglue.utils import rbd

class DetectorDnn:

    def __init__(self, detector='SUPERPOINT', device=None, processing_size=None, max_keypoints=2048, 
                 confidence_threshold=0.2, use_cache=True):
        """
        Initialize extractor (SUPERPOINT, DISK, ALIKED, SIFT, DoGHardNet) and LightGlue detector/matcher.
        
        Args:
            device: torch.device or None (auto-detect)
            processing_size: (width, height) or None (no resize) or -1 (no resize)
            max_keypoints: Maximum number of keypoints to detect
            confidence_threshold: Minimum confidence for matches
            use_cache: Enable feature caching for repeated images
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        detector = detector.upper()
        if detector == 'SUPERPOINT':
            self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)
        elif detector == 'DISK':
            self.extractor = DISK(max_num_keypoints=max_keypoints).eval().to(self.device)
            self.matcher = LightGlue(features='disk').eval().to(self.device)
        elif detector == 'ALIKED':
            self.extractor = ALIKED(max_num_keypoints=max_keypoints).eval().to(self.device)
            self.matcher = LightGlue(features='aliked').eval().to(self.device)
        elif detector == 'DOGHARDNET':
            self.extractor = DoGHardNet(max_num_keypoints=max_keypoints).eval().to(self.device)
            self.matcher = LightGlue(features='doghardnet').eval().to(self.device)
        elif detector == 'SIFT':
            self.extractor = SIFT(max_num_keypoints=max_keypoints).eval().to(self.device) 
            self.matcher = LightGlue(features='sift').eval().to(self.device)
        else:
            raise ValueError(f"Unsupported detector type: {detector}")        
            
        self.confidence_threshold = confidence_threshold
        self.use_cache = use_cache
        
        # Handle processing size
        if processing_size == -1 or processing_size is None:
            self.processing_size = None
        else:
            self.processing_size = processing_size if processing_size else (640, 480)
        
        self.info = str()
        self._keypoints1, self._keypoints2, self._matches = np.array([]), np.array([]), np.array([])
        self._feature_cache = {} if use_cache else None

    def _preprocess_image(self, frame):
        """Preprocess image to tensor"""
        if self.processing_size is not None:
            frame_resized = cv2.resize(frame, self.processing_size, interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame
        
        # LightGlue tensors: (Batch, Channel, Height, Width)
        # Convert to grayscale and enhance contrast
        if frame_resized.ndim == 3:
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_resized
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
               
        # Convert to RGB for SuperPoint
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # To tensor
        tensor = torch.from_numpy(rgb / 255.0).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def extract_features(self, frame_tensor, keypoint_threshold=0.005):
        # Extract Features (SuperPoint)
        with torch.no_grad():
            feats = self.extractor.extract(frame_tensor)            
        return feats
        
    def match_features(self, feats1, feats2):
        """Match features using LightGlue"""
        if feats1 is None or feats2 is None:
            return np.array([]), np.array([]), np.array([])
        
        try:
            with torch.no_grad():
                # Match features using LightGlue
                matches01 = self.matcher({'image0': feats1, 'image1': feats2})
                feats0, feats1, matches01 = [rbd(x) for x in [feats1, feats2, matches01]]
                matches = matches01['matches'].cpu().numpy()
                scores = matches01['scores'].cpu().numpy()
                
                valid_mask = scores > self.confidence_threshold
                valid_matches = matches[valid_mask]
                
            return feats0['keypoints'].cpu().numpy(), feats1['keypoints'].cpu().numpy(), valid_matches
        
        except Exception as e:
            print(f"Matching error: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def reset(self):
        """Reset internal state"""
        self._keypoints1, self._keypoints2, self._matches = [], [], []
    
    def run(self, image1, image2):
        """Run full detection and matching pipeline"""
        self.reset()
        times = {}
        
        # Preprocessing
        t = time.time()
        tensor1 = self._preprocess_image(image1)
        tensor2 = self._preprocess_image(image2)
        times['preprocess'] = time.time() - t
        
        # Feature extraction
        t = time.time()
        feats1 = self.extract_features(tensor1)
        feats2 = self.extract_features(tensor2)
        times['extract'] = time.time() - t
        
        # Matching
        t = time.time()
        self._keypoints1, self._keypoints2, self._matches = self.match_features(feats1, feats2)
        times['match'] = time.time() - t
        
        # Rescale keypoints to original image size
        if self._keypoints1.size > 0 and self._keypoints2.size > 0 and self.processing_size is not None:
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            self._keypoints1 = self._keypoints1 * [w1 / self.processing_size[0], h1 / self.processing_size[1]]
            self._keypoints2 = self._keypoints2 * [w2 / self.processing_size[0], h2 / self.processing_size[1]]
        
        # Build info string
        total_time = sum(times.values())
        self.info = f"Matches: {len(self._matches)} | Total: {total_time*1000:.0f}ms"
        
        return self._keypoints1, self._keypoints2, self._matches
    
    def clear_cache(self):
        """Clear feature cache"""
        if self._feature_cache is not None:
            self._feature_cache.clear()
        if self.device is not None and self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.clear_cache()
        except Exception as e:
            pass  # Silently ignore cleanup errors
        
    def get_matched_points(self):
        pts_old, pts_new = [], []
        for m in self._matches:
            pts_old.append(self._keypoints1[m[0]].astype(int))
            pts_new.append(self._keypoints2[m[1]].astype(int))
        return np.asarray(pts_old), np.asarray(pts_new)
        
if __name__ == '__main__':
    
    # 1. Open Video    
    cap = cv2.VideoCapture("data/KITTI07/image_0/%06d.png")
        
    detectorDnn = DetectorDnn(
        detector='SIFT'
        )
    
    last_frame = None
    while True:
        
        # 2. Read Frame
        ret, frame = cap.read()
        if not ret:
            break
        
        if last_frame is not None:            
            # 3. Match Features
            keypoints1, keypoints2, matches = detectorDnn.run(frame, last_frame)
        
            # 4. Visualization
            viz_img = frame.copy()
            
            for m in matches:
                pt_old = keypoints1[m[0]].astype(int)
                pt_new = keypoints2[m[1]].astype(int)
                
                if viz_img.ndim < 3 or viz_img.shape[2] < 3:
                    viz_img = cv2.cvtColor(viz_img, cv2.COLOR_GRAY2BGR)
                
                # Draw the flow line
                cv2.line(viz_img, tuple(pt_old), tuple(pt_new), (0, 255, 0), 2)
                # Draw the new point
                cv2.circle(viz_img, tuple(pt_new), 4, (0, 0, 255), -1)

            cv2.imshow('SuperPoint + LightGlue Tracker', viz_img)
        else:
            cv2.imshow('SuperPoint + LightGlue Tracker', frame)
            
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
            
        last_frame = frame.copy()
