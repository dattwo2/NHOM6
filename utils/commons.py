import io

import cv2
import numpy as np


def read_image_from_bytes(file):
    return cv2.imdecode(np.frombuffer(io.BytesIO(file).read(), np.uint8), 1)

def xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h

def save_image(filename, image):
    cv2.imwrite(filename, image)