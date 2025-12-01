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
    # No print warning here, we will handle it in initialize_sam2

# Global instance
SAM2_PREDICTOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default config/checkpoint names
# User asked for "sam2-hiera-base" model.
# In SAM2 repo, the config is usually "sam2_hiera_b+.yaml" or similar for base_plus.
# The user specifically mentioned "sam2-hiera-base".
# The closest official model is "sam2_hiera_base_plus" (b+).
# We will use environment variables to allow flexibility.
CHECKPOINT_PATH = os.getenv("SAM2_CHECKPOINT", "checkpoints/sam2_hiera_base_plus.pt")
MODEL_CFG = os.getenv("SAM2_CONFIG", "sam2_hiera_b+.yaml")

def initialize_sam2():
    global SAM2_PREDICTOR
    if SAM2_PREDICTOR is not None:
        return SAM2_PREDICTOR

    if not HAS_SAM2:
        print("Warning: SAM2 python package not found. Falling back to Otsu segmentation.")
        return None

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Warning: SAM2 checkpoint not found at {CHECKPOINT_PATH}. Falling back to Otsu segmentation.")
        return None

    try:
        # Assuming standard SAM2 usage
        # Note: build_sam2 might need absolute path to config or be in python path.
        # We assume the user installed sam2 properly.
        sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=DEVICE)
        SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
        print(f"Success: SAM2 initialized with {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Error initializing SAM2: {e}. Falling back to Otsu segmentation.")
        return None

    return SAM2_PREDICTOR

def fallback_segmentation_otsu(image: np.ndarray) -> np.ndarray:
    """
    Robust Otsu thresholding as a fallback for body segmentation.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Otsu thresholding
    # Determine if background is light or dark to use INV or not
    median_val = np.median(gray)
    if median_val > 127:
        # Light background -> Dark object -> INV
        threshold_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        # Dark background -> Light object -> Normal
        threshold_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU

    _, mask = cv2.threshold(gray, 0, 255, threshold_type)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(mask)

    c = max(cnts, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [c], -1, 255, -1)

    return clean_mask

def segment_tadpole_sam2(image: np.ndarray) -> np.ndarray:
    """
    Segment the full tadpole (head + body + tail) using SAM2.
    Input: RGB/BGR image (numpy array)
    Output: 0/255 binary mask
    """
    predictor = initialize_sam2()

    # Fallback if SAM2 is not ready
    if predictor is None:
        return fallback_segmentation_otsu(image)

    # SAM2 expects RGB
    if image.ndim == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    predictor.set_image(image_rgb)

    h, w = image.shape[:2]

    # Improved Prompting: Use Otsu threshold to find the centroid of the "object"
    try:
        # Use our fallback logic just to find the centroid!
        # This is efficient because we need a prompt anyway.
        pre_mask = fallback_segmentation_otsu(image)

        cnts, _ = cv2.findContours(pre_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                input_point = np.array([[cx, cy]])
            else:
                x, y, w_box, h_box = cv2.boundingRect(c)
                input_point = np.array([[x + w_box/2, y + h_box/2]])
        else:
            input_point = np.array([[w/2, h/2]])

    except Exception as e:
        print(f"Error calculating prompt point: {e}. Fallback to center.")
        input_point = np.array([[w/2, h/2]])

    input_label = np.array([1]) # 1 is foreground

    # SAM2 prediction
    try:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        # Pick the mask with the highest score
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        binary_mask = (best_mask * 255).astype(np.uint8)
        return binary_mask

    except Exception as e:
        print(f"Error during SAM2 inference: {e}. Fallback to Otsu.")
        return fallback_segmentation_otsu(image)
