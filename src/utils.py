import cv2
import numpy as np
from typing import Optional

def read_image_with_unicode(path: str) -> Optional[np.ndarray]:
    """
    Reads an image from a path, handling special unicode characters.
    This is the recommended way to read images in OpenCV when file paths
    may contain non-ASCII characters.
    """
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None
