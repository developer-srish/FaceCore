"""
Utility helpers shared across easyface modules.
"""

import os
import numpy as np
from PIL import Image


def load_image_rgb(path_or_array):
    """
    Load an image and return it as an RGB numpy array (H, W, 3).
    Accepts a file path or an already-loaded numpy array / PIL Image.
    """
    if isinstance(path_or_array, np.ndarray):
        return path_or_array

    if isinstance(path_or_array, Image.Image):
        return np.array(path_or_array.convert("RGB"))

    if isinstance(path_or_array, str):
        if not os.path.exists(path_or_array):
            raise FileNotFoundError(f"Image not found: {path_or_array}")
        img = Image.open(path_or_array).convert("RGB")
        return np.array(img)

    raise TypeError("Expected a file path, PIL Image, or numpy array.")


def to_bgr(rgb_array):
    """Convert RGB numpy array to BGR (used by OpenCV internals)."""
    return rgb_array[:, :, ::-1].copy()


def to_rgb(bgr_array):
    """Convert BGR numpy array to RGB."""
    return bgr_array[:, :, ::-1].copy()


def save_image(array_rgb, output_path):
    """Save an RGB numpy array to disk using Pillow."""
    img = Image.fromarray(array_rgb.astype("uint8"), "RGB")
    img.save(output_path)
    return output_path


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two embedding vectors, scaled to 0-100."""
    a = np.asarray(vec_a).flatten()
    b = np.asarray(vec_b).flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    return round((sim + 1) / 2 * 100, 2)


def euclidean_to_score(vec_a, vec_b, scale=0.6):
    """
    dlib/face_recognition embeddings compare better with euclidean distance
    than cosine. Maps distance -> a friendly 0-100 score (0 dist = 100,
    `scale` dist ~= the usual "different person" cutoff -> low score).
    """
    a = np.asarray(vec_a).flatten()
    b = np.asarray(vec_b).flatten()
    dist = float(np.linalg.norm(a - b))
    score = max(0.0, 1.0 - dist / (2 * scale)) * 100
    return round(score, 2)


def align_face(rgb_image, landmarks_5pt, output_size=128):
    """
    Similarity-transform alignment used by swap/recognition models trained
    on the classic 5-point (2x left eye, right eye, nose, mouth-left,
    mouth-right) ArcFace-style template. Pure numpy/OpenCV.
    """
    import cv2

    src = np.array(landmarks_5pt, dtype=np.float32)

    ref = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32) * (output_size / 112.0)

    M, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
    if M is None:
        M = np.eye(2, 3, dtype=np.float32)

    aligned = cv2.warpAffine(to_bgr(rgb_image), M, (output_size, output_size), borderValue=0.0)
    return to_rgb(aligned), M