"""
Age & gender analysis via a standalone ONNX model. Uses the well-known
"age_googlenet.onnx" / "gender_googlenet.onnx" pair (or any compatible
ONNX age/gender models you point it at), run through onnxruntime.
Emotion is bolted on separately via EmotionDetector.
"""

import os
import numpy as np
import cv2
from .emotion import EmotionDetector
from .utils import load_image_rgb

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_AGE_MODEL = os.path.expanduser("~/.easyface/models/age_googlenet.onnx")
DEFAULT_GENDER_MODEL = os.path.expanduser("~/.easyface/models/gender_googlenet.onnx")

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LABELS = ["Male", "Female"]


class Analyzer:
    def __init__(self, detector, age_model_path=None, gender_model_path=None, emotion_model_path=None):
        self._detector = detector
        self._emotion = EmotionDetector(model_path=emotion_model_path)
        self._age_model_path = age_model_path or DEFAULT_AGE_MODEL
        self._gender_model_path = gender_model_path or DEFAULT_GENDER_MODEL
        self._age_session = None
        self._gender_session = None

    def _load(self, path):
        if ort is None:
            raise ImportError(
                "onnxruntime is required for age/gender analysis. "
                "Install with: uv add onnxruntime"
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Age/gender model not found at {path}. "
                "Download the model and place it there, or pass "
                "age_model_path=/gender_model_path=... to Analyzer()/Face()."
            )
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def _preprocess(self, rgb_crop):
        bgr = rgb_crop[:, :, ::-1]
        resized = cv2.resize(bgr, (224, 224)).astype(np.float32)
        mean = np.array([104.0, 177.0, 123.0], dtype=np.float32)
        resized -= mean
        tensor = resized.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
        return tensor

    def analyze(self, image, include_emotion=True):
        """
        Return age/gender (and optionally emotion) info for detected face(s).

        {
            "age": str,                  # bucketed range, e.g. "(25-32)"
            "gender": "Male" | "Female",
            "confidence": float,         # detection confidence
            "emotion": "happy",
            "emotion_scores": {...}
        }

        Returns None if no face is detected.
        Returns a list of these dicts if multiple faces are detected.
        """
        faces = self._detector.detect(image)
        if not faces:
            return None

        rgb = load_image_rgb(image)

        infos = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            crop = rgb[y1:y2, x1:x2]

            info = {"age": None, "gender": None, "confidence": f["confidence"]}

            if crop.size != 0:
                try:
                    if self._age_session is None:
                        self._age_session = self._load(self._age_model_path)
                    if self._gender_session is None:
                        self._gender_session = self._load(self._gender_model_path)

                    tensor = self._preprocess(crop)

                    age_name = self._age_session.get_inputs()[0].name
                    age_out = self._age_session.run(None, {age_name: tensor})[0][0]
                    info["age"] = AGE_BUCKETS[int(np.argmax(age_out))]

                    gender_name = self._gender_session.get_inputs()[0].name
                    gender_out = self._gender_session.run(None, {gender_name: tensor})[0][0]
                    info["gender"] = GENDER_LABELS[int(np.argmax(gender_out))]
                except (ImportError, FileNotFoundError) as e:
                    info["age_gender_error"] = str(e)

            if include_emotion:
                try:
                    emo = self._emotion.detect(image, bbox=f["bbox"])
                    if emo:
                        info["emotion"] = emo["emotion"]
                        info["emotion_scores"] = emo["scores"]
                except ImportError as e:
                    info["emotion"] = None
                    info["emotion_error"] = str(e)
                except FileNotFoundError as e:
                    info["emotion"] = None
                    info["emotion_error"] = str(e)

            infos.append(info)

        return infos[0] if len(infos) == 1 else infos