"""
Face detection wrapper around the modern MediaPipe Tasks API
(FaceLandmarker). Replaces the removed legacy mp.solutions API.
Returns simple, easy-to-read dictionaries instead of raw model objects.
"""

import os
import numpy as np
from .utils import load_image_rgb

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None
    mp_python = None
    mp_vision = None

DEFAULT_MODEL_PATH = os.path.expanduser("~/.easyface/models/face_landmarker.task")


class Detector:
    def __init__(self, model_path=None):
        if mp is None:
            raise ImportError(
                "mediapipe is required. Install with: uv add mediapipe"
            )

        self._model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {self._model_path}. "
                "Download 'face_landmarker.task' from "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task "
                "and place it there, or pass model_path=... to Detector()."
            )

        base_options = mp_python.BaseOptions(model_asset_path=self._model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=10,
            min_face_detection_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # indices into the 478-point mesh matching the classic 5-point layout
        self._mesh_5pt_idx = {
            "left_eye": 33,
            "right_eye": 263,
            "nose": 1,
            "mouth_left": 61,
            "mouth_right": 291,
        }

    def detect(self, image, min_confidence=0.5):
        """
        Detect faces in an image.

        Returns a list of dicts:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "landmarks": [[x, y], ...],   # 5-point landmarks (eyes/nose/mouth)
                "embedding": None,
                "_raw": None
            }
        """
        rgb = load_image_rgb(image)
        h, w = rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        results = []
        if not result.face_landmarks:
            return results

        for face_landmarks in result.face_landmarks:
            xs = [lm.x * w for lm in face_landmarks]
            ys = [lm.y * h for lm in face_landmarks]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            landmarks = [
                [face_landmarks[self._mesh_5pt_idx["left_eye"]].x * w,
                 face_landmarks[self._mesh_5pt_idx["left_eye"]].y * h],
                [face_landmarks[self._mesh_5pt_idx["right_eye"]].x * w,
                 face_landmarks[self._mesh_5pt_idx["right_eye"]].y * h],
                [face_landmarks[self._mesh_5pt_idx["nose"]].x * w,
                 face_landmarks[self._mesh_5pt_idx["nose"]].y * h],
                [face_landmarks[self._mesh_5pt_idx["mouth_left"]].x * w,
                 face_landmarks[self._mesh_5pt_idx["mouth_left"]].y * h],
                [face_landmarks[self._mesh_5pt_idx["mouth_right"]].x * w,
                 face_landmarks[self._mesh_5pt_idx["mouth_right"]].y * h],
            ]

            results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": 1.0,  # FaceLandmarker doesn't expose a per-face score
                "landmarks": landmarks,
                "embedding": None,
                "_raw": None,
            })

        return results

    def largest_face(self, image):
        """Convenience: return the single largest face detected, or None."""
        faces = self.detect(image)
        if not faces:
            return None

        def area(face):
            x1, y1, x2, y2 = face["bbox"]
            return (x2 - x1) * (y2 - y1)

        return max(faces, key=area)