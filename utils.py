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
    """Convert RGB numpy array to BGR (used by OpenCV / insightface internals)."""
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
    # cosine sim is -1..1 -> map to a friendlier 0..100 score
    return round((sim + 1) / 2 * 100, 2)