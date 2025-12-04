import cv2
import numpy as np
import os
import sys
import argparse
from typing import Tuple, Optional

# --- GESTION ROBUSTE DES IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from utils import read_image_with_unicode
    from segmentation_body import segment_tadpole_sam2
    from eyes_ilastik import detect_eyes_ilastik
except ImportError as e:
    print(f"❌ Erreur critique d'importation : {e}")
    sys.exit(1)

# --- CONFIGURATION ---
DEFAULT_ILASTIK_PATH = r"C:\Program Files\ilastik-1.4.1.post1-gpu\ilastik.exe"
DEFAULT_PROJECT_NAME = "eyes.ilp"

def find_ilastik_project():
    """Cherche le fichier .ilp intelligemment."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, "models", DEFAULT_PROJECT_NAME),
        os.path.join(base_dir, DEFAULT_PROJECT_NAME),
        os.path.join(current_dir, DEFAULT_PROJECT_NAME),
        "eyes.ilp"
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return None

def draw_scientific_arrow(img, pt1, pt2, color, text="", thickness=2):
    """Dessine une flèche de mesure technique."""
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
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
    ilastik_binary_path: Optional[str] = None,
    ilastik_project_path: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], float, float, float, str, str]:
    
    # 0. CONFIGURATION
    if not ilastik_binary_path: ilastik_binary_path = DEFAULT_ILASTIK_PATH
    if not ilastik_project_path: ilastik_project_path = find_ilastik_project()

    if not os.path.exists(image_path):
        return None, 0.0, 0.0, 0.0, "File not found", "unknown"

    img = read_image_with_unicode(image_path)
    if img is None:
        return None, 0.0, 0.0, 0.0, "Image unreadable", "unknown"

    output_img = img.copy()
    if debug and output_dir: os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. SEGMENTATION CORPS (SAM2)
    # ------------------------------------------------------------------
    body_length_px = 0.0
    box_points = None
    
    try:
        if segment_tadpole_sam2:
            mask_body = segment_tadpole_sam2(img)
        else:
            raise ImportError("SAM2 module missing")
        
        if cv2.countNonZero(mask_body) == 0:
            return output_img, 0.0, 0.0, 0.0, "Body not detected", "unknown"
            
        cnts, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c_max = max(cnts, key=cv2.contourArea)
            hull = cv2.convexHull(c_max)
            cv2.drawContours(output_img, [hull], -1, (0, 255, 0), 2) # CORPS EN VERT
            
            # Mesure Longueur
            rect = cv2.minAreaRect(hull) 
            box = cv2.boxPoints(rect)
            box_points = np.int32(box)
            body_length_px = float(max(rect[1]))
            
            # Flèche Longueur (Cyan)
            d1 = np.linalg.norm(box_points[0]-box_points[1])
            d2 = np.linalg.norm(box_points[1]-box_points[2])
            
            if d1 > d2: p_s, p_e = box_points[0], box_points[1]
            else:       p_s, p_e = box_points[1], box_points[2]

            draw_scientific_arrow(output_img, p_s, p_e, (255, 255, 0), f"L={int(body_length_px)}")
        
    except Exception as e:
        print(f"SAM2 Error: {e}")
        return output_img, 0.0, 0.0, 0.0, f"SAM2 Crash: {e}", "unknown"

    # ------------------------------------------------------------------
    # 2. DÉTECTION YEUX (ILASTIK)
    # ------------------------------------------------------------------
    snout_dist_px = 0.0
    eye_dist_px = 0.0
    num_eyes = 0
    eye1, eye2 = None, None
    status_eyes = "Init"
    
    try:
        eye1, eye2, eye_dist_px, status_eyes, num_eyes = detect_eyes_ilastik(
            img, mask_body, ilastik_binary_path, ilastik_project_path
        )
    except Exception as e:
        # Fallback compatibilité
        try:
            eye1, eye2, eye_dist_px, status_eyes = detect_eyes_ilastik(
                img, mask_body, ilastik_binary_path, ilastik_project_path
            )
            num_eyes = 2 if eye_dist_px > 0 else 0
        except:
            num_eyes = 0
            status_eyes = "Error"

    # ------------------------------------------------------------------
    # 3. ANALYSE ET DESSIN FINAL
    # ------------------------------------------------------------------
    if num_eyes >= 2 and eye_dist_px > 0 and box_points is not None:
        orientation = "dorsal"
        status_final = "Success"
        
        # A. DESSIN YEUX (Rouge)
        mid_eyes = ((eye1[0] + eye2[0]) / 2, (eye1[1] + eye2[1]) / 2)
        cv2.circle(output_img, eye1, 5, (0, 0, 255), -1)
        cv2.circle(output_img, eye2, 5, (0, 0, 255), -1)
        cv2.line(output_img, eye1, eye2, (0, 0, 255), 1)
        draw_scientific_arrow(output_img, eye1, eye2, (0, 0, 255), f"Yeux={int(eye_dist_px)}")
        
        # B. DESSIN NEZ (Plaquode) - ROSE/MAGENTA (pour visibilité)
        d01 = np.linalg.norm(box_points[0]-box_points[1])
        d12 = np.linalg.norm(box_points[1]-box_points[2])
        
        # On cherche les milieux des petits côtés (largeurs)
        if d01 < d12:
            end1 = (box_points[0] + box_points[1]) / 2
            end2 = (box_points[2] + box_points[3]) / 2
        else:
            end1 = (box_points[1] + box_points[2]) / 2
            end2 = (box_points[3] + box_points[0]) / 2
            
        dist1 = np.linalg.norm(np.array(mid_eyes) - np.array(end1))
        dist2 = np.linalg.norm(np.array(mid_eyes) - np.array(end2))
        
        if dist1 < dist2:
            snout_pt = end1
            snout_dist_px = dist1
        else:
            snout_pt = end2
            snout_dist_px = dist2
            
        # DESSIN EN ROSE (255, 0, 255)
        draw_scientific_arrow(output_img, mid_eyes, snout_pt, (255, 0, 255), f"Nez={int(snout_dist_px)}")

    elif num_eyes == 1:
        orientation = "profile"
        status_final = "Profile View (1 eye)"
    else:
        orientation = "ventral"
        status_final = f"Ventral/Error ({num_eyes} eyes)"

    cv2.putText(output_img, f"View: {orientation.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    if debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, "final_output.jpg"), output_img)

    return output_img, body_length_px, eye_dist_px, snout_dist_px, status_final, orientation

if __name__ == "__main__":
    # ... (Code test inchangé) ...
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs='?')
    parser.add_argument("--ilastik-path")
    parser.add_argument("--ilastik-project")
    args = parser.parse_args()
    
    target = args.image
    if not target:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = os.path.join(base, "data", "training")
        if os.path.exists(d): 
            f_list = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png'))]
            if f_list: target = os.path.join(d, f_list[0])
            
    if target:
        print(f"Analyzing {target}...")
        res, l, e, n, s, o = analyze_tadpole_microscope(
            target, debug=True, output_dir="debug_test",
            ilastik_binary_path=args.ilastik_path, ilastik_project_path=args.ilastik_project
        )
        if res is not None:
            cv2.imwrite("test_final.jpg", res)
            print(f"✅ Saved test_final.jpg")
            print(f"📊 Length: {l:.1f} | Eyes: {e:.1f} | Snout: {n:.1f} | Ori: {o} | Status: {s}")
    else:
        print("❌ No image found")