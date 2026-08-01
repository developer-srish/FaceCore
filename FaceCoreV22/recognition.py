"""
Face comparison & simple named-person recognition.

Uses dlib's ResNet face-recognition model directly (no face_recognition
package wrapper) for 128-d embeddings. Recognition works against a
small in-memory "gallery" that you can register faces into with
Face.register(name, image).
"""

import os
import numpy as np
from .utils import load_image_rgb, euclidean_to_score

try:
    import dlib
except ImportError:
    dlib = None

DEFAULT_REC_MODEL = os.path.expanduser("~/.easyface/models/dlib_face_recognition_resnet_model_v1.dat")
DEFAULT_SHAPE_MODEL = os.path.expanduser("~/.easyface/models/shape_predictor_5_face_landmarks.dat")


class Recognizer:
    def __init__(self, detector, rec_model_path=None, shape_model_path=None):
        self._detector = detector
        self._gallery = {}  # name -> embedding
        self._rec_model_path = rec_model_path or DEFAULT_REC_MODEL
        self._shape_model_path = shape_model_path or DEFAULT_SHAPE_MODEL
        self._rec_model = None
        self._shape_predictor = None

    def _load_models(self):
        if self._rec_model is not None:
            return

        if dlib is None:
            raise ImportError(
                "dlib is required for recognition/comparison. "
                "Install with: uv add dlib-bin"
            )

        if not os.path.exists(self._rec_model_path):
            raise FileNotFoundError(
                f"Recognition model not found at {self._rec_model_path}. "
                "Download 'dlib_face_recognition_resnet_model_v1.dat' from "
                "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 "
                "and place it there, or pass rec_model_path=... to Recognizer()."
            )
        if not os.path.exists(self._shape_model_path):
            raise FileNotFoundError(
                f"Shape predictor not found at {self._shape_model_path}. "
                "Download 'shape_predictor_5_face_landmarks.dat' from "
                "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 "
                "and place it there, or pass shape_model_path=... to Recognizer()."
            )

        self._rec_model = dlib.face_recognition_model_v1(self._rec_model_path)
        self._shape_predictor = dlib.shape_predictor(self._shape_model_path)

    def _embed(self, image, bbox):
        """Compute a 128-d embedding for the face at `bbox` in `image`."""
        self._load_models()
        rgb = load_image_rgb(image)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        rect = dlib.rectangle(max(0, x1), max(0, y1), x2, y2)

        shape = self._shape_predictor(rgb, rect)
        embedding = self._rec_model.compute_face_descriptor(rgb, shape)
        return np.array(embedding)

    def register(self, name, image):
        """Add a named person to the recognition gallery."""
        face = self._detector.largest_face(image)
        if face is None:
            raise ValueError(f"No face found in image for '{name}'")
        embedding = self._embed(image, face["bbox"])
        if embedding is None:
            raise ValueError(f"Could not compute embedding for '{name}'")
        self._gallery[name] = embedding

    def compare(self, image_a, image_b):
        """Return a 0-100 similarity score between the primary faces in two images."""
        face_a = self._detector.largest_face(image_a)
        face_b = self._detector.largest_face(image_b)

        if face_a is None or face_b is None:
            raise ValueError("Could not detect a face in one or both images.")

        emb_a = self._embed(image_a, face_a["bbox"])
        emb_b = self._embed(image_b, face_b["bbox"])

        return euclidean_to_score(emb_a, emb_b)

    def recognize(self, image, threshold=65.0):
        """
        Compare the primary face in `image` against everyone in the gallery.
        Returns the best-matching name, or "Unknown" if nothing clears the threshold.
        """
        if not self._gallery:
            return "Unknown"

        face = self._detector.largest_face(image)
        if face is None:
            return "Unknown"

        embedding = self._embed(image, face["bbox"])

        best_name, best_score = "Unknown", -1.0
        for name, gal_embedding in self._gallery.items():
            score = euclidean_to_score(embedding, gal_embedding)
            if score > best_score:
                best_name, best_score = name, score

        return best_name if best_score >= threshold else "Unknown"