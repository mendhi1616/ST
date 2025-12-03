import os
import sys
import numpy as np
import cv2
import torch

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False

SAM2_PREDICTOR = None


def initialize_sam2():
    """Initialise SAM2 via Hugging Face"""
    global SAM2_PREDICTOR
    if SAM2_PREDICTOR is not None:
        return SAM2_PREDICTOR

    if not HAS_SAM2:
        print("Warning: SAM2 python package not found.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # On utilise le modèle base-plus qui est un bon compromis
        print(f"Loading SAM2 model on {device}...")
        predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus")
        predictor.model.to(device)
        SAM2_PREDICTOR = predictor
        return SAM2_PREDICTOR
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
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

def robust_segmentation_redness(image: np.ndarray) -> np.ndarray:
    """
    Segmentation de secours basée sur la couleur (Fond Rouge vs Têtard Gris).
    Utilisée pour générer la boite guide pour SAM.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    b, g, r = cv2.split(image)

    # Indice de Rougeur : le fond est très rouge, le têtard peu.
    redness = cv2.subtract(r, ((g.astype(np.float32) + b.astype(np.float32))/2).astype(np.uint8))

    # Seuillage Otsu : Fond (rouge) >> Seuil >> Têtard (gris)
    # Le fond sera Blanc (255), le têtard Noir (0)
    _, mask_bg = cv2.threshold(redness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # On inverse : Têtard = Blanc (255)
    mask_body = cv2.bitwise_not(mask_bg)

    # Nettoyage pour avoir une forme propre
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel, iterations=4)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel, iterations=2)

    # Garder le plus gros objet centré
    cnts, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask_body)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(final_mask, [c], -1, 255, -1)

    return final_mask

def get_bright_spots(image: np.ndarray, mask_roi: np.ndarray) -> list:
    """
    Trouve les points très brillants (reflets) pour les donner comme points NÉGATIFS à SAM.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # On cherche les pixels très blancs (>240) à l'intérieur de la zone d'intérêt
    _, mask_bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    mask_bright = cv2.bitwise_and(mask_bright, mask_bright, mask=mask_roi)

    cnts, _ = cv2.findContours(mask_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    negative_points = []
    for c in cnts:
        # On prend le centre du reflet
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            negative_points.append([cx, cy])

    return negative_points


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
    Segmentation SAM2 avec correction automatique d'inversion.
    """
    predictor = initialize_sam2()

    # Fallback si SAM2 absent
    if predictor is None:
        return robust_segmentation_redness(image)

    # Préparation image
    if image.ndim == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    try:
        predictor.set_image(image_rgb)

        # 1. Génération du guide (Prompt) via la méthode Rougeur
        pre_mask = robust_segmentation_redness(image)
        cnts, _ = cv2.findContours(pre_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_img, w_img = image.shape[:2]

        # Initialisation par défaut pour éviter NameError ou None
        input_point = np.array([[w_img // 2, h_img // 2]], dtype=np.float32)
        input_label = np.array([1], dtype=np.int32)
        box_prompt = None

        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # Boîte englobante avec une petite marge
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            box_prompt = np.array([x1, y1, x2, y2], dtype=np.float32)

            # Point central positif
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                input_point = np.array([[cx, cy]], dtype=np.float32)
                input_label = np.array([1], dtype=np.int32)
            else:
                 # Fallback to bbox center if moments fail
                 cx = x + w // 2
                 cy = y + h // 2
                 input_point = np.array([[cx, cy]], dtype=np.float32)
                 input_label = np.array([1], dtype=np.int32)
        else:
            # Fallback already handled by default initialization
            pass

        # 2. Prédiction SAM
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=box_prompt,
            multimask_output=True,
        )

        # 3. Sélection du meilleur masque
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]
        final_mask = (best_mask > 0).astype(np.uint8) * 255

        # --- 4. CORRECTION AUTOMATIQUE D'INVERSION (CORNER CHECK) ---
        # On vérifie les 4 coins de l'image. S'ils sont blancs, c'est que le masque est inversé.
        corners = [
            final_mask[0, 0],
            final_mask[0, w_img-1],
            final_mask[h_img-1, 0],
            final_mask[h_img-1, w_img-1]
        ]
        # Si plus de 2 coins sont "sélectionnés" (255), c'est le fond !
        if sum([1 for c in corners if c > 127]) > 2:
            if debug: print("[SAM2] Masque inversé détecté (Fond sélectionné). Inversion...")
            final_mask = cv2.bitwise_not(final_mask)

        # Nettoyage final
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        # Garder le plus gros objet restant (pour virer les îlots de bruit du fond)
        cnts_final, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(final_mask)
        if cnts_final:
            c_max = max(cnts_final, key=cv2.contourArea)
            cv2.drawContours(clean_mask, [c_max], -1, 255, -1)

        return clean_mask

    except Exception as e:
        print(f"Error SAM2: {e}. Fallback Otsu.")
        return robust_segmentation_redness(image)
