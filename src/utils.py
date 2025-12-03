import cv2
import numpy as np
from typing import Optional

def read_image_with_unicode(path: str) -> Optional[np.ndarray]:
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None
