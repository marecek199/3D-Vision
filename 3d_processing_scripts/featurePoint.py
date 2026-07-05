import torch
import cv2
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

def run_deep_tracker(video_source=0):
    # 1. Setup Device (GPU is highly recommended)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # 2. Load Models
    # SuperPoint: The Feature Extractor
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    
    # LightGlue: The Matcher (configured for SuperPoint features)
    matcher = LightGlue(features='superpoint').eval().to(device)

    cap = cv2.VideoCapture(video_source)
    
    # State variables for the "previous" frame
    last_frame_tensor = None
    last_feats = None
    last_image_np = None

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Preprocess Image
        # LightGlue expects tensors: (Batch, Channel, Height, Width)
        frame_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_tensor / 255.0).float()
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device) # Shape: (1, 3, H, W)

        # 4. Extract Features (SuperPoint)
        # Returns: keypoints, keypoint_scores, descriptors
        with torch.no_grad():
            feats = extractor.extract(frame_tensor)

        # 5. Match with Previous Frame (LightGlue)
        if last_feats is not None:
            with torch.no_grad():
                # Match current frame (feats) vs previous frame (last_feats)
                matches01 = matcher({'image0': last_feats, 'image1': feats})
                
                # Prune matches to keep only high confidence ones
                # 'rbd' removes batch dimension for easier processing
                feats0, feats1, matches01 = [rbd(x) for x in [last_feats, feats, matches01]]
                
                # Extract indices of matched points
                matches = matches01['matches']  # indices [index_in_0, index_in_1]
                scores = matches01['scores']    # confidence (0 to 1)

            # --- Visualization ---
            # Convert points to numpy for drawing
            kpts0 = feats0['keypoints'].cpu().numpy()
            kpts1 = feats1['keypoints'].cpu().numpy()
            matches = matches.cpu().numpy()
            
            valid_matches = matches[scores.cpu().numpy() > 0.5] # Threshold confidence

            # Draw matches on the current frame
            # We draw a line from where the point WAS (kpts0) to where it IS (kpts1)
            # Note: To visualize strictly on 'frame', we map kpts1. 
            # To see flow, we can draw lines.
            
            viz_img = frame.copy()
            
            for m in valid_matches:
                pt_old = kpts0[m[0]].astype(int)
                pt_new = kpts1[m[1]].astype(int)
                
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

        # 6. Update State
        last_frame_tensor = frame_tensor
        last_feats = feats
        last_image_np = frame

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_deep_tracker("data/blais.mp4")
    # run_deep_tracker("data/KITTI07/image_0/%06d.png")
