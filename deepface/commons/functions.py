import os
import base64
from pathlib import Path
from PIL import Image
import requests
import numpy as np
import cv2
import tensorflow as tf
from deprecated import deprecated
from deepface.detectors import FaceDetector
from deepface.commons.logger import Logger

logger = Logger(module="commons.functions")

# TensorFlow version compatibility
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".", maxsplit=1)[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image


# Initialize storage folders
def initialize_folder():
    home = get_deepface_home()
    deepface_home_path = os.path.join(home, ".deepface")
    weights_path = os.path.join(deepface_home_path, "weights")

    os.makedirs(deepface_home_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)
    logger.info(f"Initialized directories at {deepface_home_path}")


def get_deepface_home():
    return str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))


# Load image helpers
def load_base64_img(uri):
    """Load image from base64 string."""
    encoded_data = uri.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_bgr


def load_image(img):
    """Load image from path, URL, base64 string, or numpy array."""
    if isinstance(img, np.ndarray):
        return img, None

    if isinstance(img, Path):
        img = str(img)

    if img.startswith("data:image/"):
        return load_base64_img(img), None

    if img.startswith("http"):
        response = requests.get(img, stream=True, timeout=60)
        img_bgr = np.array(Image.open(response.raw).convert("RGB"))[:, :, ::-1]
        return img_bgr, img

    if not os.path.isfile(img):
        raise ValueError(f"File {img} does not exist")

    if not img.isascii():
        raise ValueError(f"Image path must not contain non-ASCII characters: {img}")

    img_bgr = cv2.imread(img)
    return img_bgr, img


# Face extraction and processing
import cv2
import numpy as np
from keras.preprocessing import image
from deepface.commons import functions

def extract_faces(
    img,
    target_size=(224, 224),
    detector_backend="opencv",
    grayscale=False,
    enforce_detection=True,
    align=True,
):
    """Extract faces from an image."""
    extracted_faces = []

    # Load image từ đường dẫn hoặc các định dạng khác
    img, img_name = functions.load_image(img)

    # Kiểm tra img sau khi load
    if not isinstance(img, np.ndarray):
        raise TypeError(
            f"Giá trị img không phải là NumPy array sau khi load: {type(img)}"
        )

    img_region = [0, 0, img.shape[1], img.shape[0]]

    # Xử lý phát hiện khuôn mặt
    if detector_backend == "skip":
        face_objs = [(img, img_region, 1.0)]
    else:
        # Sử dụng model detector từ deepface
        face_detector = functions.FaceDetector.build_model(detector_backend)
        face_objs = functions.FaceDetector.detect_faces(face_detector, detector_backend, img, align)

    # Kiểm tra nếu không tìm thấy khuôn mặt
    if len(face_objs) == 0 and enforce_detection:
        if img_name:
            raise ValueError(
                f"Không thể phát hiện khuôn mặt trong {img_name}. "
                "Hãy kiểm tra ảnh hoặc đặt enforce_detection=False."
            )
        else:
            raise ValueError(
                "Không thể phát hiện khuôn mặt. "
                "Hãy kiểm tra ảnh hoặc đặt enforce_detection=False."
            )

    # Nếu không phát hiện được khuôn mặt, nhưng enforce_detection=False
    if len(face_objs) == 0 and not enforce_detection:
        face_objs = [(img, img_region, 1.0)]

    for current_img, current_region, confidence in face_objs:
        # Kiểm tra current_img có phải là NumPy array
        if not isinstance(current_img, np.ndarray):
            raise TypeError(
                f"Dữ liệu current_img không hợp lệ: {type(current_img)}"
            )

        if current_img.shape[0] > 0 and current_img.shape[1] > 0:
            if grayscale:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            # Resize và padding ảnh
            factor_0 = target_size[0] / current_img.shape[0]
            factor_1 = target_size[1] / current_img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (
                int(current_img.shape[1] * factor),
                int(current_img.shape[0] * factor),
            )
            current_img = cv2.resize(current_img, dsize)

            diff_0 = target_size[0] - current_img.shape[0]
            diff_1 = target_size[1] - current_img.shape[1]
            if not grayscale:
                current_img = np.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                        (0, 0),
                    ),
                    "constant",
                )
            else:
                current_img = np.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                    ),
                    "constant",
                )

            # Đảm bảo ảnh đạt kích thước target_size
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

            # Chuẩn hóa giá trị pixel
            img_pixels = image.img_to_array(current_img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            region_obj = {
                "x": int(current_region[0]),
                "y": int(current_region[1]),
                "w": int(current_region[2]),
                "h": int(current_region[3]),
            }

            extracted_face = [img_pixels, region_obj, confidence]
            extracted_faces.append(extracted_face)

    if len(extracted_faces) == 0 and enforce_detection:
        raise ValueError(
            f"Khuôn mặt được phát hiện không hợp lệ. Kích thước ảnh là {img.shape}. "
            "Hãy đặt enforce_detection=False nếu muốn xử lý ảnh không có khuôn mặt."
        )

    return extracted_faces




def normalize_input(img, normalization="base"):
    """Normalize image according to specified method."""
    img *= 255
    if normalization == "Facenet":
        return (img - img.mean()) / img.std()
    if normalization == "Facenet2018":
        return img / 127.5 - 1
    if normalization == "VGGFace":
        img[..., 0] -= 93.594
        img[..., 1] -= 104.762
        img[..., 2] -= 129.186
    if normalization == "ArcFace":
        return (img - 127.5) / 128
    return img


def find_target_size(model_name):
    target_sizes = {
        "VGG-Face": (224, 224), "Facenet": (160, 160), "ArcFace": (112, 112)
    }
    return target_sizes.get(model_name, None)
