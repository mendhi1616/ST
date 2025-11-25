import pickle
import os
from typing import Any

IMREAD_COLOR = 1
COLOR_BGR2GRAY = 0

# Simplified image represented as nested lists

def imread(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def imwrite(path: str, img: Any):
    try:
        with open(path, 'wb') as f:
            pickle.dump(img, f)
        return True
    except Exception:
        return False


def imdecode(stream, flag=None):
    return imread(stream)


def ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness):
    return img


def circle(img, center, radius, color, thickness):
    return img


def cvtColor(img, code):
    return img


def threshold(img, thresh, maxval, type):
    return thresh, img


def morphologyEx(img, op, kernel, iterations=1):
    return img


def findContours(img, mode, method):
    return [], None


def contourArea(contour):
    return 0


def fitEllipse(contour):
    return (0, 0), (0, 0), 0


def minAreaRect(contour):
    return ((0, 0), (0, 0), 0)


def drawContours(img, contours, idx, color, thickness):
    return img


def bitwise_and(src1, src2, mask=None):
    return src1


def arcLength(contour, closed):
    return 0


def moments(contour):
    return {"m00": 1, "m10": 1, "m01": 1}


def line(img, pt1, pt2, color, thickness):
    return img


def putText(img, text, org, fontFace, fontScale, color, thickness):
    return img

FONT_HERSHEY_SIMPLEX = 0
THRESH_BINARY_INV = 0
THRESH_OTSU = 0
RETR_EXTERNAL = 0
CHAIN_APPROX_SIMPLE = 0
MORPH_OPEN = 0
