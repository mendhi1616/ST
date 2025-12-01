import cv2
import numpy as np
import math
import os
from typing import Optional, Tuple, List
from utils import read_image_with_unicode


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

    Args:
        image_path: chemin de l'image.
        debug: si True, sauvegarde des images intermédiaires dans output_dir.
        output_dir: dossier pour les images de debug.

    Returns:
        (image_annotée, longueur_corps_px, distance_yeux_px, statut)
    """

    # ------------------------------------------------------------------
    # 0) Lecture de l'image
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
    # 1) SEGMENTATION DU CORPS : suppression du fond rouge (HSV)
    # ------------------------------------------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Masque du ROUGE (fond) : H dans [0,10] U [170,180]
    # S et V suffisamment élevés
    mask1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Corps = ce qui n'est PAS rouge
    mask_body = cv2.bitwise_not(mask_red)

    # Nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug:
        cv2.imwrite(os.path.join(output_dir, "mask_body_raw.png"), mask_body)

    # On garde uniquement le plus gros objet (le têtard)
    cnts, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return output_img, 0.0, 0.0, "Aucun contour détecté"

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 500:  # trop petit = bruit
        return output_img, 0.0, 0.0, "Contour trop petit"

    # Convex hull pour lisser la forme
    hull = cv2.convexHull(cnt)
    mask_hull = np.zeros_like(mask_body)
    cv2.drawContours(mask_hull, [hull], -1, 255, -1)

    cv2.drawContours(output_img, [hull], -1, (0, 255, 0), 2)

    if debug:
        cv2.imwrite(os.path.join(output_dir, "mask_hull.png"), mask_hull)

    # ------------------------------------------------------------------
    # 2) LONGUEUR DU CORPS : boîte englobante minimum
    # ------------------------------------------------------------------
    rect = cv2.minAreaRect(hull)   # ((cx,cy), (w,h), angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.drawContours(output_img, [box], 0, (0, 255, 255), 2)

    (cx_rect, cy_rect), (w_rect, h_rect), angle = rect
    body_length_px = float(max(w_rect, h_rect))

    # On trace une flèche approximative de longueur (entre les 2 points de la boîte les plus éloignés)
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
    # 3) DÉTECTION ROBUSTE DES YEUX (dark blob detection dans la tête)
    # ------------------------------------------------------------------
    # Passage en LAB pour travailler sur la luminance
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # On ne garde que la luminance à l'intérieur du hull
    L_body = cv2.bitwise_and(L, L, mask=mask_hull)
    L_body[mask_hull == 0] = 255  # fond blanc (pour ne pas le considérer sombre)

    # Boîte englobante du corps
    x, y, w_box, h_box = cv2.boundingRect(hull)

    # On définit la "tête" comme le tiers avant du corps
    head_ratio = 0.33
    if w_box >= h_box:
        # corps plutôt horizontal → tête dans le tiers gauche
        head_width = max(10, int(w_box * head_ratio))
        head_roi = L_body[y : y + h_box, x : x + head_width]
        head_offset_x = x
        head_offset_y = y
    else:
        # corps plutôt vertical → tête dans le tiers haut
        head_height = max(10, int(h_box * head_ratio))
        head_roi = L_body[y : y + head_height, x : x + w_box]
        head_offset_x = x
        head_offset_y = y

    if debug:
        cv2.imwrite(os.path.join(output_dir, "head_roi.png"), head_roi)

    # On floute pour réduire le bruit
    blur = cv2.GaussianBlur(head_roi, (9, 9), 0)

    # Seuillage adaptatif pour extraire les taches sombres (yeux)
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )

    # Nettoyage et petites taches
    kernel_eye = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_eye, iterations=2)

    if debug:
        cv2.imwrite(os.path.join(output_dir, "head_threshold.png"), th)

    cnts_eyes, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    eye_candidates: List[Tuple[int, int, int]] = []  # (cx, cy, luminance_L)
    for c in cnts_eyes:
        area_c = cv2.contourArea(c)
        if 20 < area_c < 1000:  # intervalle de taille plausible pour un œil
            ex, ey, ew, eh = cv2.boundingRect(c)
            cx_eye = ex + ew // 2
            cy_eye = ey + eh // 2
            # on regarde la valeur de luminance (plus elle est faible, plus c'est sombre)
            intensity = int(head_roi[cy_eye, cx_eye])
            eye_candidates.append((cx_eye, cy_eye, intensity))

    # On garde les 2 plus sombres
    eye_candidates = sorted(eye_candidates, key=lambda k: k[2])  # tri par L (0 = noir)
    eye_candidates = eye_candidates[:2]

    eye_centers: List[Tuple[int, int]] = []
    for (cx_eye, cy_eye, _) in eye_candidates:
        gx = cx_eye + head_offset_x
        gy = cy_eye + head_offset_y
        eye_centers.append((gx, gy))
        cv2.circle(output_img, (gx, gy), 5, (255, 255, 0), -1)

    eye_distance_px = 0.0
    status_msg = "OK"

    if len(eye_centers) == 2:
        eye_distance_px = float(np.linalg.norm(
            np.array(eye_centers[0]) - np.array(eye_centers[1])
        ))
        cv2.line(output_img, eye_centers[0], eye_centers[1], (255, 255, 255), 2)
        cv2.putText(
            output_img,
            f"D={int(eye_distance_px)}",
            (int((eye_centers[0][0] + eye_centers[1][0]) / 2),
             int((eye_centers[0][1] + eye_centers[1][1]) / 2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    else:
        status_msg = "Yeux non détectés de manière fiable"
        eye_distance_px = 0.0

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