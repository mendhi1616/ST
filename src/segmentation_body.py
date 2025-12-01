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

    # If SAM2 is not available (e.g. in test env without GPU/libs), return empty or fallback?
    # User said "SAM2 is the only body segmentation method".
    # So if it fails, we probably should return a failed mask (zeros).
    if predictor is None:
        # For testing purposes in this environment, if SAM2 is missing, we might want to mock it
        # or return a dummy mask if we are testing integration.
        # But for production, it should be zeros.
        # I will print a warning.
        print("SAM2 predictor not available.")
        return np.zeros(image.shape[:2], dtype=np.uint8)

    # SAM2 expects RGB
    if image.ndim == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    predictor.set_image(image_rgb)

    # We need a prompt for SAM2. The user said "covering the whole tadpole".
    # If we don't have a prompt, maybe we can use a center point or a box covering the image?
    # Usually we need at least one point.
    # Let's try to find a rough center of mass or use the center of the image as a positive point.
    # Or maybe a grid of points?
    # The user didn't specify the prompt strategy.
    # A simple strategy: Use the center of the image. Tadpoles are usually centered-ish.
    # Better: Use a simple threshold to find "something" and put a point in it.
    # But user said "Remove all previous segmentation methods... No Otsu".
    # So we should rely on SAM2 capabilities.
    # Maybe we can prompt with the center point?

    h, w = image.shape[:2]
    input_point = np.array([[w/2, h/2]])
    input_label = np.array([1]) # 1 is foreground

    # SAM2 prediction
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True # We might want to see multiple and pick best
    )

    # Pick the mask with the highest score
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]

    # Convert to uint8 0/255
    binary_mask = (best_mask * 255).astype(np.uint8)

    return binary_mask
