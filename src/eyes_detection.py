import cv2
import numpy as np
import math
import os
import sys
import argparse
from typing import Tuple, Optional, List, Dict, Any

from utils import read_image_with_unicode
from segmentation_body import segment_tadpole_sam2
from eyes_ilastik import detect_eyes_ilastik
from orientation import classify_orientation


def analyze_tadpole_microscope(
    image_path: str,
    debug: bool = False,
    output_dir: Optional[str] = None,
    ilastik_binary_path: Optional[str] = None,
    ilastik_project_path: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], float, float, str, str]:
    """
    Analyzes a tadpole image using SAM2 for body segmentation, Ilastik for eye detection,
    and a CNN for orientation classification.

    Returns: (annotated_image, body_length_px, eye_distance_px, status, orientation)
    """

    # ------------------------------------------------------------------
    # 0) Read Image
    # ------------------------------------------------------------------
    if not os.path.exists(image_path):
        return None, 0.0, 0.0, "File not found", "unknown"

    img = read_image_with_unicode(image_path)
    if img is None:
        return None, 0.0, 0.0, "Image unreadable", "unknown"

    output_img = img.copy()

    if debug:
        if output_dir is None:
            output_dir = "debug_output"
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) SEGMENTATION DU CORPS (SAM2)
    # ------------------------------------------------------------------
    mask_body = segment_tadpole_sam2(img)

    if debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, "mask_body_sam2.png"), mask_body)

    # Check if mask is empty
    if cv2.countNonZero(mask_body) == 0:
        return output_img, 0.0, 0.0, "Body not detected (SAM2 failed)", "unknown"

    # Find contours to get hull and bounding rect
    cnts, _ = cv2.findContours(mask_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return output_img, 0.0, 0.0, "No contours in mask", "unknown"

    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)

    cv2.drawContours(output_img, [hull], -1, (0, 255, 0), 2)

    # ------------------------------------------------------------------
    # 1.5) ORIENTATION CLASSIFICATION
    # ------------------------------------------------------------------
    # Crop the image to the bounding box of the body for classification
    x, y, w, h = cv2.boundingRect(mask_body)
    # Add some padding
    pad = int(0.1 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)

    cropped_body = img[y0:y1, x0:x1]

    orientation, orientation_conf = classify_orientation(cropped_body)

    cv2.putText(
        output_img,
        f"Ori: {orientation} ({orientation_conf:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    # ------------------------------------------------------------------
    # 2) BODY LENGTH: minAreaRect
    # ------------------------------------------------------------------
    rect = cv2.minAreaRect(hull)   # ((cx,cy), (w,h), angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.drawContours(output_img, [box], 0, (0, 255, 255), 2)

    (cx_rect, cy_rect), (w_rect, h_rect), angle = rect
    body_length_px = float(max(w_rect, h_rect))

    # Draw length arrow
    # Identify the two furthest points in the box to draw the arrow
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

    # ------------------------------------------------------------------
    # 3) EYE DETECTION (Ilastik)
    # ------------------------------------------------------------------
    # valeurs par défaut (au cas où on ne trouve pas 2 yeux)
    eye1 = None
    eye2 = None
    eye_dist_px = 0.0
    snout_dist_px = 0.0
    status_eyes = "Not computed"

    # >>> PAS de condition sur l’orientation ici <<<
    eye1, eye2, eye_dist_px, status_eyes = detect_eyes_ilastik(
        img,
        mask_body,
        ilastik_binary_path=ilastik_binary_path,
        ilastik_project_path=ilastik_project_path,
    )

    MIN_EYE_DIST_FRAC = 0.15  # 15% de la longueur du corps
    min_eye_dist_px = MIN_EYE_DIST_FRAC * body_length_px

    if eye1 is not None and eye2 is not None and eye_dist_px < min_eye_dist_px:
        status_eyes = f"Rejected_too_close_{eye_dist_px:.1f}px"
        eye1 = None
        eye2 = None
        eye_dist_px = 0.0


    # On ne trace D et EN QUE si on a bien 2 yeux
    if eye1 is not None and eye2 is not None and status_eyes == "Success":
        # --- dessin des yeux + D ---
        cv2.circle(output_img, eye1, 5, (255, 255, 0), -1)
        cv2.circle(output_img, eye2, 5, (255, 255, 0), -1)
        cv2.line(output_img, eye1, eye2, (255, 255, 255), 2)

        eye_mid = (
            int((eye1[0] + eye2[0]) / 2),
            int((eye1[1] + eye2[1]) / 2),
        )
        eye_mid_np = np.array(eye_mid, dtype=np.float32)

        cv2.putText(
            output_img,
            f"D={int(eye_dist_px)}",
            (eye_mid[0], eye_mid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # --- calcul de EN : perpendiculaire, côté le plus court ---
        eye_vec = np.array(eye2, dtype=np.float32) - np.array(eye1, dtype=np.float32)
        if np.linalg.norm(eye_vec) > 1e-6:
            eye_vec = eye_vec / np.linalg.norm(eye_vec)
            perp = np.array([-eye_vec[1], eye_vec[0]], dtype=np.float32)

            h, w = mask_body.shape[:2]
            max_step = int(np.hypot(h, w))

            def raycast(dir_vec):
                last_valid = None
                for s in range(1, max_step):
                    x = int(round(eye_mid[0] + dir_vec[0] * s))
                    y = int(round(eye_mid[1] + dir_vec[1] * s))

                    if x < 0 or x >= w or y < 0 or y >= h:
                        break
                    if mask_body[y, x] > 0:
                        last_valid = (x, y)
                    else:
                        break
                if last_valid is None:
                    return None, 0.0
                dist = float(
                    np.linalg.norm(
                        np.array(last_valid, dtype=np.float32) - eye_mid_np
                    )
                )
                return last_valid, dist

            p1, d1 = raycast(perp)
            p2, d2 = raycast(-perp)

            candidates = []
            if p1 is not None and d1 > 0:
                candidates.append((p1, d1))
            if p2 is not None and d2 > 0:
                candidates.append((p2, d2))

            if candidates:
                snout_point, snout_dist_px = min(candidates, key=lambda x: x[1])

                cv2.circle(output_img, snout_point, 5, (0, 0, 255), -1)
                cv2.line(output_img, eye_mid, snout_point, (255, 255, 255), 2)

                label_pos = (
                    int((eye_mid[0] + snout_point[0]) / 2),
                    int((eye_mid[1] + snout_point[1]) / 2) - 10,
                )
                cv2.putText(
                    output_img,
                    f"EN={int(snout_dist_px)}",
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )



    if debug and output_dir:
        cv2.imwrite(os.path.join(output_dir, "final_output.png"), output_img)

    return output_img, body_length_px, eye_dist_px, snout_dist_px, status_eyes, orientation



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test eyes detection on a single image.")
    parser.add_argument("image_path", nargs='?', help="Path to the input image")
    parser.add_argument("--ilastik-path", help="Path to Ilastik binary")
    parser.add_argument("--ilastik-project", help="Path to Ilastik project file")
    args = parser.parse_args()

    # Default path for quick testing if no arg provided
    target_path = args.image_path
    if not target_path:
        # Try to find a default file or prompt
        print("No image path provided. Please provide a path as an argument.")
        sys.exit(1)

    print(f"Analyzing {target_path}...")

    annotated, length, eyes, snout, status, ori = analyze_tadpole_microscope(
        target_path,
        debug=True,
        output_dir="debug_test",
        ilastik_binary_path=args.ilastik_path,
        ilastik_project_path=args.ilastik_project
    )

    if annotated is not None:
        output_path = "test_result_segmentation.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"\n✅ Result saved to: {os.path.abspath(output_path)}")
        print(f"Debug images saved to: {os.path.abspath('debug_test')}")
        print(f"📊 Metrics:")
        print(f"   - Body Length: {length} px")
        print(f"   - Eye Distance: {eyes} px")
        print(f"   - Orientation: {ori}")
        print(f"   - Status: {status}")
    else:
        print("❌ Analysis failed to produce an image.")
        print(f"   - Status: {status}")
