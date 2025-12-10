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
    global SAM2_PREDICTOR
    if SAM2_PREDICTOR is not None: return SAM2_PREDICTOR
    if not HAS_SAM2: return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        print(f"Loading SAM2 on {device}...")
        predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus")
        predictor.model.to(device)
        SAM2_PREDICTOR = predictor
        return SAM2_PREDICTOR
    except Exception as e:
        print(f"Error loading SAM2: {e}")
        return None

def robust_segmentation_redness(image: np.ndarray) -> np.ndarray:
    """
    Segmentation 'Rougeur' pour trouver la boite englobante.
    Le fond est ROUGE, le têtard est GRIS/NOIR.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    b, g, r = cv2.split(image)
    
    # Indice : Rouge - Moyenne(Vert, Bleu)
    redness = cv2.subtract(r, ((g.astype(np.float32) + b.astype(np.float32))/2).astype(np.uint8))
    
    # Le fond est rouge (clair), le têtard est sombre.
    # Otsu va séparer les deux.
    _, mask_bg = cv2.threshold(redness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # On inverse : Têtard = Blanc
    mask_body = cv2.bitwise_not(mask_bg)
    
    # Gros nettoyage pour avoir une "patate" propre qui servira de boite
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    # On garde le plus gros objet
    cnts, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask_body)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(final_mask, [c], -1, 255, -1)
        
    return final_mask

def postprocess_body_mask(mask: np.ndarray, pre_mask: np.ndarray = None) -> np.ndarray:
    # mask = sortie brute de SAM (0/1 ou 0/255)
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    # Si on a aussi la rougeur, on la prépare
    if pre_mask is not None:
        if pre_mask.ndim == 3:
            pre_mask = pre_mask.squeeze()
        pre_u8 = (pre_mask > 0).astype(np.uint8) * 255
    else:
        pre_u8 = None

    # 1) Dilation pour "épaissir" un peu le masque SAM
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_dil = cv2.dilate(mask_u8, kernel_big, iterations=1)

    # 2) Si on a la rougeur : on coupe dans l’enveloppe rouge
    if pre_u8 is not None:
        combined = cv2.bitwise_and(mask_dil, pre_u8)
    else:
        combined = mask_dil

    # 3) Petit nettoyage
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)

    # 4) On garde le plus gros blob
    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(mask_clean)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(final, [c], -1, 255, -1)

    return final



def segment_tadpole_sam2(image: np.ndarray, debug=False) -> np.ndarray:
    """
    Segment avec SAM2 guidé par une BOÎTE (calculée via la rougeur).
    """
    predictor = initialize_sam2()
    
    # 1. Pré-calcul de la boite via la méthode Rougeur (très robuste sur tes images)
    pre_mask = robust_segmentation_redness(image)
    
    # Si SAM2 n'est pas là, on renvoie directement le masque rougeur (c'est déjà mieux qu'Otsu)
    if predictor is None:
        return pre_mask

    # Préparation SAM2
    if image.ndim == 3: image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else: image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    predictor.set_image(image_rgb)

    try:
        cnts, _ = cv2.findContours(pre_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        box_prompt = None
        input_point = None
        input_label = None
        
        h, w = image.shape[:2]

        if cnts:
            c = max(cnts, key=cv2.contourArea)
            
            # --- A. BOÎTE ENGLOBANTE (LE SECRET) ---
            x, y, wb, hb = cv2.boundingRect(c)
            pad = 10 # Marge de sécurité
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + wb + pad)
            y2 = min(h, y + hb + pad)
            box_prompt = np.array([x1, y1, x2, y2], dtype=np.float32)
            
            # --- B. POINT CENTRAL ---
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                input_point = np.array([[cx, cy]], dtype=np.float32)
                input_label = np.array([1], dtype=np.int32)
        else:
            # Fallback centre
            input_point = np.array([[w//2, h//2]], dtype=np.float32)
            input_label = np.array([1], dtype=np.int32)

        if debug: print(f"SAM2 Box: {box_prompt}")

        # 3. Prédiction avec BOÎTE + POINT
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=box_prompt,
            multimask_output=True,
        )

        best_mask = masks[np.argmax(scores)]
        return postprocess_body_mask(best_mask, pre_mask=pre_mask)

    except Exception as e:
        print(f"SAM2 Error: {e}. Fallback Redness.")
        return pre_mask