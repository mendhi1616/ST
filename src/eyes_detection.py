import cv2
import numpy as np
import math
import os
import sys
from typing import Tuple, Optional, List, Dict, Any
from utils import read_image_with_unicode

def analyze_tadpole_microscope(image_path: str, debug: bool = False, output_dir: Optional[str] = None) -> Tuple[Optional[np.ndarray], float, float, str]:
    """
    Analyzes a tadpole image to detect body length and interocular distance.

    Args:
        image_path: Path to the image file.
        debug: If True, saves debug images.
        output_dir: Directory to save debug images (if debug is True).

    Returns:
        Tuple containing:
            - processed_image (np.ndarray or None): Image with annotations.
            - body_length_px (float): Length of the body in pixels.
            - eye_distance_px (float): Distance between eyes in pixels.
            - status_msg (str): Status message ("Succès", "Echec", etc).
    """

    # Handle output directory for debug images
    if debug and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(image_path):
        return None, 0.0, 0.0, "Fichier introuvable"

    img = read_image_with_unicode(image_path)
    if img is None:
        return None, 0.0, 0.0, "Image illisible"

    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. BODY DETECTION
    # Otsu's thresholding inverted
    _, mask_body = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening to remove noise
    kernel = np.ones((5,5), np.uint8)
    mask_body = cv2.morphologyEx(mask_body, cv2.MORPH_OPEN, kernel, iterations=2)

    contours_body, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_body:
        return None, 0.0, 0.0, "Echec corps"

    # Assume the largest contour is the body
    c_body = max(contours_body, key=cv2.contourArea)

    # Measure Length
    if len(c_body) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(c_body)
        body_length_px = max(MA, ma)
    else:
        rect = cv2.minAreaRect(c_body)
        body_length_px = max(rect[1])

    cv2.drawContours(output_img, [c_body], -1, (0, 255, 0), 2) # Body in GREEN

    # 2. EYE DETECTION (ADAPTIVE STRATEGY)
    # ROI: Search only inside the body mask
    body_only = cv2.bitwise_and(gray, gray, mask=mask_body)
    # Set background to white (255) so dark eyes stand out
    body_only[mask_body == 0] = 255

    # Strict threshold for eyes (dark spots)
    _, mask_eyes = cv2.threshold(body_only, 65, 255, cv2.THRESH_BINARY_INV)
    contours_eyes, _ = cv2.findContours(mask_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # -- FILTERING --
    candidats_parfaits = [] # Circular
    candidats_moyens = []   # Not circular but good size

    for c in contours_eyes:
        area = cv2.contourArea(c)
        if 10 < area < 600: # Broad size filter
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue

            circularity = 4 * math.pi * area / (perimeter * perimeter)

            # Debug drawing (Orange = candidate)
            cv2.drawContours(output_img, [c], -1, (0, 165, 255), 1)

            if circularity > 0.5: # Circularity > 0.5
                candidats_parfaits.append(c)
            else:
                candidats_moyens.append(c)

    # SELECTION STRATEGY
    selection = []
    mode = ""

    # Sort by area (largest to smallest)
    candidats_parfaits.sort(key=cv2.contourArea, reverse=True)
    candidats_moyens.sort(key=cv2.contourArea, reverse=True)

    if debug:
        print(f"DEBUG: Body Length: {body_length_px}")
        print(f"DEBUG: Found {len(contours_eyes)} eye contours initially.")
        print(f"DEBUG: Perfect candidates: {len(candidats_parfaits)}")
        print(f"DEBUG: Medium candidates: {len(candidats_moyens)}")

    if len(candidats_parfaits) >= 2:
        selection = candidats_parfaits[:2]
        mode = "Precise (Ronds)"
    elif len(candidats_parfaits) == 1 and len(candidats_moyens) >= 1:
        selection = [candidats_parfaits[0], candidats_moyens[0]]
        mode = "Mixte"
    elif len(candidats_moyens) >= 2:
        selection = candidats_moyens[:2]
        mode = "Fallback (Non-ronds)"

    eye_distance_px = 0.0
    status_msg = "Yeux HS"

    if len(selection) == 2:
        oeil_1 = selection[0]
        oeil_2 = selection[1]

        M1 = cv2.moments(oeil_1)
        M2 = cv2.moments(oeil_2)

        if M1["m00"] != 0 and M2["m00"] != 0:
            c1 = (int(M1["m10"]/M1["m00"]), int(M1["m01"]/M1["m00"]))
            c2 = (int(M2["m10"]/M2["m00"]), int(M2["m01"]/M2["m00"]))

            dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

            # SAFETY CHECK: Distance must be small (< 25% of body length)
            # This prevents detecting the gut or other dark spots far apart
            ratio_dist_corps = dist / body_length_px

            if ratio_dist_corps < 0.25:
                eye_distance_px = dist
                # Final drawing in RED
                cv2.line(output_img, c1, c2, (0, 0, 255), 2)
                cv2.drawContours(output_img, selection, -1, (0, 0, 255), -1)
                status_msg = f"Succès ({mode})"
            else:
                status_msg = f"Rejet (Ecart trop grand: {int(dist)}px)"
                # Draw in PURPLE to show rejection
                cv2.line(output_img, c1, c2, (255, 0, 255), 2)
        else:
             status_msg = "Echec Moments"

    cv2.putText(output_img, f"L: {int(body_length_px)} | Yeux: {int(eye_distance_px)}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, "debug_analysis.jpg"), output_img)
        cv2.imwrite(os.path.join(output_dir, "debug_body_mask.jpg"), mask_body)
        cv2.imwrite(os.path.join(output_dir, "debug_eyes_mask.jpg"), mask_eyes)

    return output_img, body_length_px, eye_distance_px, status_msg

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Analyzing {path}...")
        img, l, e, s = analyze_tadpole_microscope(path, debug=True, output_dir="debug_output")
        print(f"Status: {s}")
        print(f"Body: {l:.2f} px")
        print(f"Eyes: {e:.2f} px")
    else:
        print("Usage: python eyes_detection.py <image_path>")
