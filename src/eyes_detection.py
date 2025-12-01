import cv2
import numpy as np
import math
import os
import sys
from typing import Tuple, Optional, List, Dict, Any
from utils import read_image_with_unicode

HAS_SAM = False
SAM_PREDICTOR = None

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor  
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    SamPredictor = None  


def detect_eyes_hough(
    img_bgr: np.ndarray,
    mask_body: np.ndarray,
    hull: np.ndarray,
    main_axis: np.ndarray,
    body_length_px: float,
    debug: bool = False,
    debug_dir: Optional[str] = None,
):
    """
    Nouvelle détection des yeux :
    - restreinte à la zone 'tête' (extrémité du corps)
    - HoughCircles sur la luminance
    Retourne : (points, distance_px, statut)
    """

    h_img, w_img = img_bgr.shape[:2]

    # 1) PCA déjà calculée en amont : main_axis (2D), body_length_px
    hull_pts = hull.reshape(-1, 2).astype(np.float32)
    mean_hull = np.mean(hull_pts, axis=0)
    cov = np.cov(hull_pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]

    proj = (hull_pts - mean_hull) @ main_axis
    t_min, t_max = proj.min(), proj.max()


    # On choisit l'extrémité "tête" = t_max (arbitrairement)
    # zone tête = dernier 25% de la longueur projetée
    head_start = t_max - 0.25 * axis_len

    # 2) On fabrique un masque "tête" à partir des points du hull dont la projection > head_start
    head_mask = np.zeros_like(mask_body)
    for p in hull_pts:
        t = float((p - mean_hull) @ main_axis)
        if t >= head_start:
            cv2.circle(head_mask, (int(p[0]), int(p[1])), 3, 255, -1)

    head_mask = cv2.dilate(head_mask, np.ones((9, 9), np.uint8), iterations=2)
    head_mask = cv2.bitwise_and(head_mask, mask_body)

    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "head_mask.png"), head_mask)

    # 3) Luminance dans Lab + CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.bitwise_and(L, L, mask=head_mask)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)

    # on inverse pour que les yeux (sombres) deviennent forts
    L_inv = cv2.bitwise_not(L_eq)
    blur = cv2.medianBlur(L_inv, 5)

    # 4) HoughCircles dans la zone tête
    # rayon approximatif en fonction de la taille du corps
    r_min = max(2, int(0.005 * body_length_px))
    r_max = max(r_min + 2, int(0.04 * body_length_px))

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=0.05 * body_length_px,
        param1=100,
        param2=10,
        minRadius=r_min,
        maxRadius=r_max,
    )

    if circles is None or len(circles[0]) < 2:
        return None, 0.0, "Yeux non détectés (Hough)"

    circles = np.round(circles[0, :]).astype("int")  # (x, y, r)

    # 5) filtrer et choisir la meilleure paire
    best_pair = None
    best_score = 1e9
    eye_distance_px = 0.0

    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]

            # distance entre centres
            dx, dy = x2 - x1, y2 - y1
            dist = math.hypot(dx, dy)
            if dist < 0.02 * axis_len or dist > 0.4 * axis_len:
                continue

            # perpendicularité avec l'axe du corps
            eyes_angle = math.degrees(math.atan2(dy, dx))
            body_angle = math.degrees(math.atan2(main_axis[1], main_axis[0]))
            diff = abs((eyes_angle - body_angle + 180) % 180 - 90)  # 0 = parfait

            if diff > 40:
                continue

            # contraste : moyenne dans le disque vs autour
            mask_eye = np.zeros((h_img, w_img), np.uint8)
            cv2.circle(mask_eye, (x1, y1), r1, 255, -1)
            cv2.circle(mask_eye, (x2, y2), r2, 255, -1)
            mean_in = cv2.mean(L_eq, mask_eye)[0]

            mask_ring = cv2.dilate(mask_eye, np.ones((5, 5), np.uint8), 1)
            mask_ring = cv2.subtract(mask_ring, mask_eye)
            mean_out = cv2.mean(L_eq, mask_ring)[0]
            contrast = mean_out - mean_in  # yeux plus sombres -> contrast > 0

            # score combiné: angle + (1/contraste)
            if contrast <= 0:
                continue
            score = diff + 20.0 / contrast

            if score < best_score:
                best_score = score
                best_pair = ((x1, y1), (x2, y2))
                eye_distance_px = dist

    if best_pair is None:
        return None, 0.0, "Yeux non détectés (filtrage)"

    return best_pair, float(eye_distance_px), "OK (Hough)"


def get_sam_predictor() -> Optional["SamPredictor"]:
    """
    Initialise SAM une seule fois si :
    - les libs sont installées
    - le checkpoint existe
    - un device (GPU/CPU) est dispo

    Retourne None si SAM n'est pas utilisable (on fera alors un fallback Otsu).
    """
    global SAM_PREDICTOR

    if not HAS_SAM:
        return None

    if SAM_PREDICTOR is not None:
        return SAM_PREDICTOR

    checkpoint = os.getenv("SAM_CHECKPOINT_PATH", "sam_vit_h_4b8939.pth")
    if not os.path.exists(checkpoint):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to(device=device)

    SAM_PREDICTOR = SamPredictor(sam)
    return SAM_PREDICTOR

def get_body_mask_auto(
    img_bgr: np.ndarray,
    debug: bool = False,
    debug_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Retourne un masque binaire du têtard (corps) de la même taille que l'image.

    - Étape 1 : segmentation grossière par Otsu pour trouver où est le têtard
    - Étape 2 : on en déduit automatiquement une bounding box autour du têtard
    - Étape 3 : si SAM est dispo + checkpoint trouvé -> on donne cette box à SAM
    - Sinon : on utilise directement le masque Otsu comme fallback
    """

    h, w = img_bgr.shape[:2]

    # --- 1) Masque grossier Otsu ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask_otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Rien trouvé -> on renvoie juste le masque Otsu (ou tout noir)
        return mask_otsu

    # On prend le plus gros contour comme têtard
    c = max(cnts, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(c)

    # On slightly pad la box pour être sûr d'englober tout le têtard
    pad = int(0.05 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + w_box + pad)
    y1 = min(h, y + h_box + pad)

    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        dbg = img_bgr.copy()
        cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "box_otsu_prompt.png"), dbg)

    # --- 2) Essayer SAM avec cette box comme prompt ---
    predictor = get_sam_predictor()
    if predictor is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        box_np = np.array([[x0, y0, x1, y1]], dtype=np.float32)  # shape (1, 4)

        masks, scores, _ = predictor.predict(
            box=box_np,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        mask_sam = masks[best_idx].astype("uint8") * 255

        if debug and debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "body_mask_sam.png"), mask_sam)

        return mask_sam

    # --- 3) Fallback : pas de SAM -> on utilise Otsu ---
    if debug and debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "body_mask_otsu.png"), mask_otsu)

    return mask_otsu



def imread_windows_special(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except: return None

def draw_scientific_arrow(img, pt1, pt2, color, text="", thickness=2):
    cv2.line(img, pt1, pt2, color, thickness)
    cv2.arrowedLine(img, pt1, pt2, color, thickness, tipLength=0.05)
    cv2.arrowedLine(img, pt2, pt1, color, thickness, tipLength=0.05)
    if text:
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (mid_x - 2, mid_y - h - 5), (mid_x + w + 2, mid_y + 5), (0, 0, 0), -1)
        cv2.putText(img, text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def analyze_tadpole_microscope(
    image_path: str,
    debug: bool = False,
    output_dir: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], float, float, str]:
    """
    Analyse une image de têtard pour :
    - segmenter le corps
    - estimer la longueur du corps (en pixels)
    - détecter la distance inter-oculaire (en pixels)

    Retourne : (image_annotée, longueur_corps_px, distance_yeux_px, statut)
    """

    # ------------------------------------------------------------------
    # 0) Lecture
    # ------------------------------------------------------------------
    if not os.path.exists(image_path):
        return None, 0.0, 0.0, "Fichier introuvable"

    img = read_image_with_unicode(image_path)
    if img is None:
        return None, 0.0, 0.0, "Image illisible"

    h_img, w_img = img.shape[:2]
    output_img = img.copy()

    if debug:
        if output_dir is None:
            output_dir = "debug_output"
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) SEGMENTATION DU CORPS (SAM + Otsu si dispo, sinon HSV rouge)
    # ------------------------------------------------------------------
    try:
        mask_body = get_body_mask_auto(img, debug=debug, debug_dir=output_dir)
    except NameError:
        mask_body = None

    if mask_body is None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # fond rouge
        mask1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
        mask_red = cv2.bitwise_or(mask1, mask2)
        # corps = pas rouge
        mask_body = cv2.bitwise_not(mask_red)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug:
        cv2.imwrite(os.path.join(output_dir, "mask_body.png"), mask_body)

    cnts, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return output_img, 0.0, 0.0, "Aucun contour détecté"

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 500:
        return output_img, 0.0, 0.0, "Contour trop petit"

    hull = cv2.convexHull(cnt)
    mask_hull = np.zeros_like(mask_body)
    cv2.drawContours(mask_hull, [hull], -1, 255, -1)

    cv2.drawContours(output_img, [hull], -1, (0, 255, 0), 2)

    if debug:
        cv2.imwrite(os.path.join(output_dir, "mask_hull.png"), mask_hull)

    # ------------------------------------------------------------------
    # 2) LONGUEUR DU CORPS : minAreaRect
    # ------------------------------------------------------------------
    rect = cv2.minAreaRect(hull)   # ((cx,cy), (w,h), angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.drawContours(output_img, [box], 0, (0, 255, 255), 2)

    (cx_rect, cy_rect), (w_rect, h_rect), angle = rect
    body_length_px = float(max(w_rect, h_rect))

    # flèche de longueur = deux points les plus éloignés de la box
    max_d = 0.0
    p1, p2 = box[0], box[1]
    for i in range(4):
        for j in range(i + 1, 4):
            d = float(np.linalg.norm(box[i] - box[j]))
            if d > max_d:
                max_d = d
                p1, p2 = box[i], box[j]

    cv2.arrowedLine(output_img, tuple(p1), tuple(p2), (0, 255, 255), 2, tipLength=0.03)
    cv2.putText(
        output_img,
        f"L={int(body_length_px)}",
        (min(p1[0], p2[0]), min(p1[1], p2[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    if debug:
        cv2.imwrite(os.path.join(output_dir, "body_with_box.png"), output_img)

    # ------------------------------------------------------------------
    # 3) AXE PRINCIPAL DU CORPS (PCA) -> pour Hough
    # ------------------------------------------------------------------
    hull_pts = hull.reshape(-1, 2).astype(np.float32)
    mean_hull = np.mean(hull_pts, axis=0)
    cov = np.cov(hull_pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]  # vecteur 2D

    # ------------------------------------------------------------------
    # 4) DÉTECTION DES YEUX AVEC HOUGH DANS LA ZONE TÊTE
    # ------------------------------------------------------------------
    best_pair, eye_distance_px, status_eyes = detect_eyes_hough(
        img_bgr=img,
        mask_body=mask_hull,
        hull=hull,
        main_axis=main_axis,
        body_length_px=body_length_px,
        debug=debug,
        debug_dir=output_dir,
    )

    if best_pair is not None:
        (x1, y1), (x2, y2) = best_pair
        cv2.circle(output_img, (x1, y1), 5, (255, 255, 0), -1)
        cv2.circle(output_img, (x2, y2), 5, (255, 255, 0), -1)
        cv2.line(output_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            output_img,
            f"D={int(eye_distance_px)}",
            (int((x1 + x2) / 2), int((y1 + y2) / 2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        status_msg = status_eyes
    else:
        eye_distance_px = 0.0
        status_msg = status_eyes  # "Yeux non détectés (...)"

    if debug and output_dir is not None:
        cv2.imwrite(os.path.join(output_dir, "final_output.png"), output_img)

    return output_img, body_length_px, eye_distance_px, status_msg





# =======================================================
# ZONE DE TEST CIBLÉE (DEBUG SUR UN FICHIER PRÉCIS)
# =======================================================
if __name__ == "__main__":
    # Ton dossier racine
    target_folder = r"C:\Users\User\Desktop\results\biométrie"
    
    # LE FICHIER QUE TU VEUX TESTER
    target_filename = "MC120002.JPG" 
    
    print(f"--- RECHERCHE ET TEST SUR : {target_filename} ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_debug = os.path.join(base_dir, "data", "results", f"debug_{target_filename}")
    
    found_path = None
    # On fouille tous les sous-dossiers pour trouver MC120002.JPG
    for root, dirs, files in os.walk(target_folder):
        if target_filename in files:
            found_path = os.path.join(root, target_filename)
            break
            
    if found_path:
        print(f"✅ Fichier trouvé : {found_path}")
        print("Analyse en cours...")
        
        # On active le mode debug=True pour dessiner tous les détails (contours jaunes, etc.)
        res_img, len_px, eyes_px, status = analyze_tadpole_microscope(found_path, debug=True)
        
        # Sauvegarde du résultat visuel
        if res_img is not None:
            cv2.imwrite(output_debug, res_img)
            print(f"📸 Résultat sauvegardé sous : {output_debug}")
            print(f"📊 Données brutes :")
            print(f"   - Statut : {status}")
            print(f"   - Longueur (px) : {len_px}")
            print(f"   - Distance Yeux (px) : {eyes_px}")
            print("-> Va voir l'image générée pour comprendre pourquoi la segmentation échoue !")
        else:
            print("❌ Erreur critique : L'analyse n'a rien renvoyé.")
    else:
        print(f"❌ Impossible de trouver {target_filename} dans le dossier {target_folder}")
        print("Vérifie que le fichier existe bien et que le nom est exact.")