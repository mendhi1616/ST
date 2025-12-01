import os
import subprocess
import numpy as np
import cv2
import tempfile
import uuid
from typing import Optional

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

    print(f"Running Ilastik: {' '.join(cmd)}")

    try:
        # Check if project file exists (better error message)
        if not os.path.exists(project):
            print(f"Warning: Ilastik project file not found at {project}")
            # We proceed anyway, maybe it's a relative path that works for the binary context,
            # but usually it's good to warn.

        # Run subprocess
        # Using check_call will raise CalledProcessError if return code != 0
        # If binary is not found, it raises FileNotFoundError (WinError 2)
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except FileNotFoundError:
        print(f"Error: Ilastik binary not found at '{binary}'. Please check the configuration.")
        if os.path.exists(input_path):
            os.remove(input_path)
        return None

    except Exception as e:
        print(f"Error running Ilastik: {e}")
        # Clean up
        if os.path.exists(input_path):
            os.remove(input_path)
        return None

    # 3. Read the output
    if not os.path.exists(output_path):
        print(f"Ilastik output file not found: {output_path}")
        if os.path.exists(input_path):
            os.remove(input_path)
        return None

    # Read TIFF. cv2.imread might handle multi-page tiff or multi-channel
    prob_map = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)

    # Clean up
    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(output_path):
        os.remove(output_path)

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

    # prob_map shape: (H, W, Classes)
    # We assume we need the "eye" class.
    # If the user trained 2 classes (Background, Eye), usually channel 1 is Eye.

    if prob_map.ndim == 3:
        eye_prob = prob_map[:, :, 1] # Assumption: Channel 1 is Eye
    else:
        eye_prob = prob_map

    # Normalize to 0-255 if it's float
    if eye_prob.dtype != np.uint8:
        eye_prob_u8 = (eye_prob * 255).astype(np.uint8)
    else:
        eye_prob_u8 = eye_prob

    # 2. Mask with body mask
    if body_mask.shape[:2] != eye_prob_u8.shape[:2]:
        body_mask = cv2.resize(body_mask, (eye_prob_u8.shape[1], eye_prob_u8.shape[0]))

    masked_prob = cv2.bitwise_and(eye_prob_u8, eye_prob_u8, mask=body_mask)

    # 3. Threshold
    _, thresh = cv2.threshold(masked_prob, 127, 255, cv2.THRESH_BINARY)

    # 4. Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # stats: [x, y, width, height, area]
    if num_labels < 3: # 0 is background, need at least 1 and 2
        return None, None, 0.0, "Eyes not detected (<2 components)"

    # Create list of (area, centroid) excluding background
    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        center = centroids[i]
        components.append((area, center))

    # Sort by area descending
    components.sort(key=lambda x: x[0], reverse=True)

    # Take top 2
    if len(components) < 2:
        return None, None, 0.0, "Eyes not detected (<2 valid components)"

    c1 = components[0][1] # (x, y)
    c2 = components[1][1]

    eye_center_1 = (int(c1[0]), int(c1[1]))
    eye_center_2 = (int(c2[0]), int(c2[1]))

    # 5. Compute distance
    dx = eye_center_1[0] - eye_center_2[0]
    dy = eye_center_1[1] - eye_center_2[1]
    distance_px = np.sqrt(dx*dx + dy*dy)

    return eye_center_1, eye_center_2, float(distance_px), "Success"
