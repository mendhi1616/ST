import os
import subprocess
import numpy as np
import cv2
import tempfile
import uuid
import tifffile
from typing import Optional, Tuple

# Paramètres Ilastik
DEFAULT_ILASTIK_BINARY = os.getenv("ILASTIK_PATH", "ilastik")
DEFAULT_ILASTIK_PROJECT = os.getenv("ILASTIK_PROJECT", "eyes.ilp")

def binarize_eye_prob(eye_prob: np.ndarray) -> np.ndarray:
    """Binarise la map de probabilité."""
    eye_prob = np.asarray(eye_prob)
    if eye_prob.max() <= 1.0:
        eye_prob = (eye_prob * 255).astype(np.uint8)
    else:
        eye_prob = eye_prob.astype(np.uint8)
        
    _, mask = cv2.threshold(eye_prob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def run_ilastik_headless(image: np.ndarray, binary_path: str, project_path: str) -> Optional[np.ndarray]:
    """Lance Ilastik en arrière-plan."""
    temp_dir = tempfile.gettempdir()
    input_filename = f"in_{uuid.uuid4()}.png"
    input_path = os.path.join(temp_dir, input_filename)
    output_filename = f"out_{uuid.uuid4()}.tiff"
    output_path = os.path.join(temp_dir, output_filename)

    cv2.imwrite(input_path, image)

    cmd = [
        binary_path if binary_path else DEFAULT_ILASTIK_BINARY,
        "--headless",
        f"--project={project_path if project_path else DEFAULT_ILASTIK_PROJECT}",
        "--output_format=tiff",
        "--export_source=probabilities",
        f"--output_filename_format={output_path}",
        input_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except:
        pass
    finally:
        if os.path.exists(input_path): os.remove(input_path)

    # Récupération fichier (avec gestion des suffixes potentiels)
    final_path = output_path
    if not os.path.exists(output_path):
        base_name = os.path.splitext(output_path)[0]
        candidates = [f for f in os.listdir(temp_dir) if f.startswith(os.path.basename(base_name))]
        if candidates: final_path = os.path.join(temp_dir, candidates[0])
        else: return None

    try:
        prob_map = tifffile.imread(final_path)
    except: return None
    finally:
        if os.path.exists(final_path): os.remove(final_path)

    return prob_map

def detect_eyes_ilastik(
    image: np.ndarray, 
    body_mask: np.ndarray, 
    ilastik_binary_path=None, 
    ilastik_project_path=None
) -> Tuple[Optional[tuple], Optional[tuple], float, str, int]:
    
    """
    Retourne : (pt1, pt2, distance, status, NOMBRE_YEUX_TROUVES)
    """
    
    prob_map = run_ilastik_headless(image, ilastik_binary_path, ilastik_project_path)
    if prob_map is None:
        return None, None, 0.0, "Ilastik Failed", 0

    # Sélection canal intelligent
    if prob_map.ndim == 3 and prob_map.shape[2] >= 2:
        c0, c1 = prob_map[..., 0], prob_map[..., 1]
        eye_prob = c0 if c1.max() == 0 else c1
    else:
        eye_prob = prob_map

    # Masquage par le corps
    if body_mask is not None:
        if eye_prob.shape != body_mask.shape:
            body_mask = cv2.resize(body_mask, (eye_prob.shape[1], eye_prob.shape[0]))
        eye_prob = cv2.bitwise_and(eye_prob, eye_prob, mask=body_mask)

    # Analyse des objets
    mask = binarize_eye_prob(eye_prob)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    candidates = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        # Filtre de taille pour ne pas compter le bruit comme un œil
        if area > 15: 
            candidates.append(centroids[i])
            
    num_eyes_found = len(candidates)

    # Logique de retour basée sur le nombre
    if num_eyes_found < 2:
        return None, None, 0.0, f"Found {num_eyes_found} eye(s)", num_eyes_found

    # Si >= 2, on cherche la meilleure paire (les plus écartés pour éviter les doublons)
    # Tri simple pour l'instant (on pourrait trier par taille via stats)
    c1 = candidates[0]
    c2 = candidates[1]
    
    # Si on a plus de 2 candidats, on prend les 2 plus éloignés (souvent les vrais yeux)
    max_dist = 0
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            d = np.linalg.norm(candidates[i] - candidates[j])
            if d > max_dist:
                max_dist = d
                c1, c2 = candidates[i], candidates[j]

    p1 = (int(c1[0]), int(c1[1]))
    p2 = (int(c2[0]), int(c2[1]))
    
    return p1, p2, float(max_dist), "Success", num_eyes_found