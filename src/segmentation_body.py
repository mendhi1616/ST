import os
import sys
import numpy as np
import cv2
import torch

# Try importing SAM2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False
    print("Warning: SAM2 not found. Segmentation will fail unless mocked.")

# Global instance
SAM2_PREDICTOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Using 'sam2_hiera_base.pt' as the checkpoint name based on model name 'sam2-hiera-base'
# The user said "Use the model sam2-hiera-base"
CHECKPOINT_PATH = os.getenv("SAM2_CHECKPOINT", "sam2_hiera_base.pt")
# The config file usually matches the model name in sam2 repo
MODEL_CFG = "sam2_hiera_base.yaml"

def initialize_sam2():
    global SAM2_PREDICTOR
    if SAM2_PREDICTOR is not None:
        return SAM2_PREDICTOR

    if not HAS_SAM2:
        return None

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Warning: SAM2 checkpoint not found at {CHECKPOINT_PATH}")
        return None

    try:
        # Assuming standard SAM2 usage
        sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=DEVICE)
        SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        return None

    return SAM2_PREDICTOR

def segment_tadpole_sam2(image: np.ndarray) -> np.ndarray:
    """
    Segment the full tadpole (head + body + tail) using SAM2.
    Input: RGB/BGR image (numpy array)
    Output: 0/255 binary mask
    """
    predictor = initialize_sam2()

    # If SAM2 is not available (e.g. in test env without GPU/libs), return zeros or mock
    if predictor is None:
        print("SAM2 predictor not available.")
        return np.zeros(image.shape[:2], dtype=np.uint8)

    # SAM2 expects RGB
    if image.ndim == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    predictor.set_image(image_rgb)

    h, w = image.shape[:2]

    # Improved Prompting: Use Otsu threshold to find the centroid of the "object"
    # This is more robust than assuming the tadpole is exactly in the center.
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Invert because tadpoles are usually dark on light background?
        # Or light on dark? User said "Otsu Inverse thresholding" was used before.
        # Let's assume standard Otsu on Inverted image if object is dark.
        # But to be safe, let's try to just find "something".

        # Determine if background is light or dark.
        # Simple heuristic: median pixel value.
        median_val = np.median(gray)
        if median_val > 127:
            # Light background -> Dark object
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Dark background -> Light object
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find largest contour
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                input_point = np.array([[cx, cy]])
            else:
                # Fallback to center of bounding box
                x, y, w_box, h_box = cv2.boundingRect(c)
                input_point = np.array([[x + w_box/2, y + h_box/2]])
        else:
            # Fallback to image center
            input_point = np.array([[w/2, h/2]])

    except Exception as e:
        print(f"Error calculating prompt point: {e}. Fallback to center.")
        input_point = np.array([[w/2, h/2]])

    input_label = np.array([1]) # 1 is foreground

    # SAM2 prediction
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # Pick the mask with the highest score
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]

    # Convert to uint8 0/255
    binary_mask = (best_mask * 255).astype(np.uint8)

    return binary_mask
