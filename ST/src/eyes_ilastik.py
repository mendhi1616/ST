import os
import subprocess
import numpy as np
import cv2
import tempfile
import uuid
import tifffile
from typing import Optional


def binarize_eye_prob(eye_prob: np.ndarray,
                      debug_prefix: Optional[str] = None) -> np.ndarray:
    """Convertit la carte de prob en masque binaire (0/255) avec Otsu + debug optionnel."""
    eye_prob = np.asarray(eye_prob)

    if eye_prob.dtype == np.uint8:
        eye_prob_u8 = eye_prob
    elif np.issubdtype(eye_prob.dtype, np.floating):
        eye_prob_u8 = np.clip(eye_prob * 255.0, 0, 255).astype(np.uint8)
    elif eye_prob.dtype == np.uint16:
        eye_prob_u8 = (eye_prob / 257).astype(np.uint8)  # 65535/255 ≈ 257
    else:
        min_v, max_v = float(eye_prob.min()), float(eye_prob.max())
        if max_v > min_v:
            eye_prob_u8 = ((eye_prob - min_v) * 255.0 / (max_v - min_v)).astype(np.uint8)
        else:
            eye_prob_u8 = np.zeros_like(eye_prob, dtype=np.uint8)

    # seuil d’Otsu
    _, eye_mask = cv2.threshold(
        eye_prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if debug_prefix is not None:
        cv2.imwrite(f"{debug_prefix}_eye_prob_u8.png", eye_prob_u8)
        cv2.imwrite(f"{debug_prefix}_eye_mask.png", eye_mask)
    return eye_mask


# Default fallback values
DEFAULT_ILASTIK_BINARY = os.getenv("ILASTIK_PATH", "ilastik")
DEFAULT_ILASTIK_PROJECT = os.getenv("ILASTIK_PROJECT", "eyes.ilp")


def run_ilastik_headless(
    image: np.ndarray,
    ilastik_binary_path: Optional[str] = None,
    ilastik_project_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Runs Ilastik in headless mode on the input image.
    Returns the probability map (height, width, channels) or None on failure.
    """

    # Resolve paths
    binary = ilastik_binary_path if ilastik_binary_path and ilastik_binary_path.strip() else DEFAULT_ILASTIK_BINARY
    project = ilastik_project_path if ilastik_project_path and ilastik_project_path.strip() else DEFAULT_ILASTIK_PROJECT

    # 1. Save input image to a temp file
    temp_dir = tempfile.gettempdir()
    input_filename = f"ilastik_input_{uuid.uuid4()}.png"
    input_path = os.path.join(temp_dir, input_filename)

    cv2.imwrite(input_path, image)

    output_filename = f"ilastik_output_{uuid.uuid4()}.tiff"
    output_path = os.path.join(temp_dir, output_filename)

    # 2. Construct command
    cmd = [
        binary,
        "--headless",
        f"--project={project}",
        "--export_source=probabilities",
        f"--output_filename_format={output_path}",
        input_path
    ]

    print(">>> Running Ilastik headless...")
    print(">>> Command:", " ".join(cmd))

    try:
        if not os.path.exists(project):
            print(f"Warning: Ilastik project file not found at {project}")

        # On NE cache PLUS stdout/stderr pour voir les erreurs
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        print(">>> Ilastik return code:", result.returncode)
        if result.stdout:
            print(">>> Ilastik STDOUT:")
            print(result.stdout)
        if result.stderr:
            print(">>> Ilastik STDERR:")
            print(result.stderr)

        if result.returncode != 0:
            print("Error: Ilastik returned non-zero exit code.")
            # clean input, garder éventuellement le output pour debug
            if os.path.exists(input_path):
                os.remove(input_path)
            return None

    except FileNotFoundError:
        print(f"Error: Ilastik binary not found at '{binary}'. Please check the configuration.")
        if os.path.exists(input_path):
            os.remove(input_path)
        return None

    except Exception as e:
        print(f"Error running Ilastik: {e}")
        if os.path.exists(input_path):
            os.remove(input_path)
        return None

    # 3. Read the output
    if not os.path.exists(output_path):
        print(f"Ilastik output file not found: {output_path}")
        if os.path.exists(input_path):
            os.remove(input_path)
        # NE PAS return prob_map ici, on renvoie None
        return None

    prob_map = tifffile.imread(output_path)

    # Clean up input (tu peux commenter la suppression de output si tu veux analyser le tiff à la main)
    if os.path.exists(input_path):
        os.remove(input_path)
    # if os.path.exists(output_path):
    #     os.remove(output_path)

    return prob_map


def detect_eyes_ilastik(
    image: np.ndarray,
    body_mask: np.ndarray,
    ilastik_binary_path: Optional[str] = None,
    ilastik_project_path: Optional[str] = None
) -> tuple:
    """
    Detect eyes using Ilastik probabilities masked by SAM2 body mask.
    Returns: (eye_center_1, eye_center_2, distance_px, status_string)

    eye_center is (x, y) tuple or None
    """

    # 1. Run Ilastik
    prob_map = run_ilastik_headless(image, ilastik_binary_path, ilastik_project_path)

    if prob_map is None:
        return None, None, 0.0, "Ilastik failed or binary not found"

# --- 1) choisir le canal des yeux : canal 1 ---
    if prob_map.ndim == 3:
        h, w, c = prob_map.shape
        print(f"[DEBUG] prob_map shape: {prob_map.shape}")
        if c >= 2:
            bg_prob  = prob_map[..., 0]
            eye_prob = prob_map[..., 1]
            print(f"[DEBUG] Using channel 1 as eye probability map.")
            print(f"[DEBUG] bg_prob min/max: {bg_prob.min()} / {bg_prob.max()}")
            print(f"[DEBUG] eye_prob min/max: {eye_prob.min()} / {eye_prob.max()}")
        else:
            # sécurité si jamais il n’y a qu’un canal
            eye_prob = prob_map[..., 0]
            print(f"[DEBUG] WARNING: only {c} channel(s), using channel 0 as fallback.")
    else:
        eye_prob = prob_map

    # --- 2) normalisation en uint8 (on garde ta logique) ---
    # --- 2) normalisation en uint8 ---
    if eye_prob.dtype == np.uint8:
        eye_prob_u8 = eye_prob
    elif np.issubdtype(eye_prob.dtype, np.floating):
        eye_prob_u8 = np.clip(eye_prob * 255.0, 0, 255).astype(np.uint8)
    elif eye_prob.dtype == np.uint16:
        eye_prob_u8 = (eye_prob / 257).astype(np.uint8)
    else:
        min_v, max_v = float(eye_prob.min()), float(eye_prob.max())
        if max_v > min_v:
            eye_prob_u8 = ((eye_prob - min_v) * 255.0 / (max_v - min_v)).astype(np.uint8)
        else:
            eye_prob_u8 = np.zeros_like(eye_prob, dtype=np.uint8)

    # 2. Optionnel : masque du corps (pour l’instant tu voulais toute l’image)
    if body_mask.shape[:2] != eye_prob_u8.shape[:2]:
        body_mask = cv2.resize(body_mask, (eye_prob_u8.shape[1], eye_prob_u8.shape[0]))

    # si tu veux ignorer le corps : on garde eye_prob_u8
    masked_prob = eye_prob_u8
    # si plus tard tu veux utiliser le corps :
    # masked_prob = cv2.bitwise_and(eye_prob_u8, eye_prob_u8, mask=body_mask)

    # 3. Binarisation
    debug_prefix = None  # ou "debug_eyes" si tu veux les PNG
    eye_mask = binarize_eye_prob(masked_prob, debug_prefix=debug_prefix)

    # 4. Composantes connexes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eye_mask, connectivity=8
    )

    if num_labels < 3:  # 0 = fond, il faut au moins 2 objets
        return None, None, 0.0, "Eyes not detected (<2 components)"

    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        center = centroids[i]
        components.append((area, center))

    components.sort(key=lambda x: x[0], reverse=True)

    if len(components) < 2:
        return None, None, 0.0, "Eyes not detected (<2 valid components)"

    c1 = components[0][1]
    c2 = components[1][1]

    eye_center_1 = (int(c1[0]), int(c1[1]))
    eye_center_2 = (int(c2[0]), int(c2[1]))

    dx = eye_center_1[0] - eye_center_2[0]
    dy = eye_center_1[1] - eye_center_2[1]
    distance_px = float(np.sqrt(dx * dx + dy * dy))

    return eye_center_1, eye_center_2, distance_px, "Success"

