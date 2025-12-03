import os
import sys
import numpy as np
import cv2
import torch

# Try importing SAM2
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False


# Global instance
SAM2_PREDICTOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default config/checkpoint names
# User asked for "sam2-hiera-base" model.
# In SAM2 repo, the config is usually "sam2_hiera_b+.yaml" or similar for base_plus.
# The user specifically mentioned "sam2-hiera-base".
# The closest official model is "sam2_hiera_base_plus" (b+).
# We will use environment variables to allow flexibility.
SAM2_PREDICTOR = None

def initialize_sam2():
    """
    Initialise SAM2 via Hugging Face:
    facebook/sam2-hiera-base-plus
    """
    global SAM2_PREDICTOR

    if SAM2_PREDICTOR is not None:
        return SAM2_PREDICTOR

    if not HAS_SAM2:
        print("Warning: SAM2 python package not found. Falling back to Otsu segmentation.")
        return None

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        print("Loading SAM2 from Hugging Face (facebook/sam2-hiera-base-plus)...")
        predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus")
        predictor.model.to(device)
        SAM2_PREDICTOR = predictor
        print("✅ SAM2 initialized from Hugging Face on", device)
        return SAM2_PREDICTOR
    except Exception as e:
        print(f"Error initializing SAM2 from Hugging Face: {e}. Falling back to Otsu segmentation.")
        return None


def get_sam_prompt_point(image: np.ndarray) -> tuple[int, int]:
    """
    Génère un point fiable pour SAM :
      - convertit en gris
      - Otsu
      - garde la plus grande composante
      - retourne le centroïde
    
    Si aucun contour → centre de l’image (fallback)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu rapide
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inversion si besoin (si fond clair)
    if np.mean(otsu == 255) > 0.8:   # trop blanc -> invert
        otsu = 255 - otsu

    cnts, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = gray.shape
        return (w // 2, h // 2)

    # plus gros contour
    c = max(cnts, key=cv2.contourArea)

    M = cv2.moments(c)
    if M["m00"] == 0:
        h, w = gray.shape
        return (w // 2, h // 2)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


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


def postprocess_body_mask(mask: np.ndarray,
                          min_area_ratio: float = 0.01,
                          max_area_ratio: float = 0.8) -> np.ndarray | None:
    """
    Nettoie le masque SAM2 :
      - binarisation
      - ouverture / fermeture morpho
      - garde la plus grande composante
      - vérifie que l'aire est raisonnable par rapport à l'image

    Retourne:
        - masque 0/255 propre
        - ou None si le masque est trop petit / trop gros (pour fallback Otsu)
    """
    if mask.ndim == 3:
        # SAM2 renvoie (1, H, W) ou (N, H, W) parfois
        mask = mask.squeeze()

    mask_u8 = (mask > 0).astype(np.uint8) * 255

    h, w = mask_u8.shape

    # Morpho pour lisser
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_clean = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Garder la plus grande composante
    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    final_mask = np.zeros_like(mask_clean)
    cv2.drawContours(final_mask, [c], -1, 255, -1)

    # Vérifier la taille du masque
    area = cv2.countNonZero(final_mask)
    ratio = area / float(h * w)

    if ratio < min_area_ratio or ratio > max_area_ratio:
        # Trop petit ou trop énorme → probablement faux
        print(f"[DEBUG] SAM2 mask area ratio suspicious: {ratio:.3f}")
        return None

    return final_mask


def segment_tadpole_sam2(image: np.ndarray, debug=False, debug_dir=None) -> np.ndarray:
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

    try:
        # On utilise notre fallback Otsu JUSTE pour trouver un point au centre du têtard
        pre_mask = fallback_segmentation_otsu(image)

        cnts, _ = cv2.findContours(pre_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                input_point = np.array([[cx, cy]], dtype=np.float32)
            else:
                x, y, w_box, h_box = cv2.boundingRect(c)
                cx = int(x + w_box / 2)
                cy = int(y + h_box / 2)
                input_point = np.array([[cx, cy]], dtype=np.float32)
        else:
            # Aucun contour → on tombe au centre de l'image
            input_point = np.array([[w / 2, h / 2]], dtype=np.float32)

    except Exception as e:
        print(f"Error calculating prompt point: {e}. Fallback to center.")
        input_point = np.array([[w / 2, h / 2]], dtype=np.float32)


    # SAM2 prediction
    try:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Pick best mask
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]  # [H, W] bool

        # Post-traitement du masque
        refined = postprocess_body_mask(best_mask)
        if refined is None:
            print("[DEBUG] SAM2 mask looks bad, falling back to Otsu.")
            return fallback_segmentation_otsu(image)

        return refined

    except Exception as e:
        print(f"Error during SAM2 inference: {e}. Fallback to Otsu.")
        return fallback_segmentation_otsu(image)
