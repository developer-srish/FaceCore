"""
Emotion detection via a lightweight ONNX model (FER+), run through
onnxruntime. Model input: 64x64 grayscale face crop. Output: 8 emotion
class scores.
"""

import os
import numpy as np
import cv2
from .utils import load_image_rgb

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_MODEL_PATH = os.path.expanduser("~/.easyface/models/emotion-ferplus-8.onnx")

EMOTIONS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]


class EmotionDetector:
    def __init__(self, model_path=None):
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._session = None

    def _load_model(self):
        if self._session is not None:
            return self._session

        if ort is None:
            raise ImportError(
                "onnxruntime is required for emotion detection. "
                "Install with: pip install onnxruntime"
            )

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Emotion model not found at {self._model_path}. "
                "Download 'emotion-ferplus-8.onnx' and place it there, "
                "or pass model_path=... to EmotionDetector()."
            )

        self._session = ort.InferenceSession(self._model_path, providers=["CPUExecutionProvider"])
        return self._session

    def _preprocess(self, rgb_crop):
        gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (64, 64))
        tensor = resized.astype(np.float32).reshape(1, 1, 64, 64)
        return tensor

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def detect(self, image, bbox=None):
        """
        Detect the dominant emotion in an image (or a cropped region if bbox given).

        Returns: {"emotion": str, "confidence": float, "scores": dict} or None
        """
        session = self._load_model()

        rgb = load_image_rgb(image)

        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            rgb = rgb[y1:y2, x1:x2]
            if rgb.size == 0:
                return None

        input_tensor = self._preprocess(rgb)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})

        raw_scores = outputs[0][0]
        probs = self._softmax(raw_scores)

        scores = {EMOTIONS[i]: round(float(probs[i]), 4) for i in range(len(EMOTIONS))}
        top_emotion = max(scores, key=scores.get)

        return {
            "emotion": top_emotion,
            "confidence": scores[top_emotion],
            "scores": scores,
        }