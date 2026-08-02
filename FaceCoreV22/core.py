"""
easyface/core.py

Face — the single friendly entry point that ties every module together.
Detection/landmarks via MediaPipe's Tasks API (FaceLandmarker), recognition
via dlib's ResNet model directly, age/gender + swap via standalone ONNX
models run through onnxruntime directly.
"""

from .detector import Detector
from .landmarks import Landmarks
from .analysis import Analyzer
from .recognition import Recognizer
from .swap import Swapper
from .emotion import EmotionDetector
from .webcam import WebcamRecognizer


class Face:
    def __init__(self, swap_model=None, age_model=None, gender_model=None,
                 face_model=None, emotion_model=None, arcface_model=None):
        """
        swap_model   : optional path to an inswapper_128.onnx model (needed for swap features)
        age_model    : optional path to an age ONNX model (googlenet-style)
        gender_model : optional path to a gender ONNX model (googlenet-style)
        face_model   : optional path to face_landmarker.task (needed for detection/landmarks)
        emotion_model: optional path to emotion-ferplus-8.onnx (needed for emotion detection)
        arcface_model: optional path to an ArcFace recognition ONNX model, e.g.
                       'w600k_r50.onnx' (needed for swap/webcam_swap -- inswapper
                       requires ArcFace-style embeddings specifically, separate
                       from the dlib model used for compare()/recognize())
        """
        self._detector = Detector(model_path=face_model)
        self._landmarks = Landmarks(self._detector)
        self._emotion_detector = EmotionDetector(model_path=emotion_model)
        self._analyzer = Analyzer(self._detector, age_model_path=age_model, gender_model_path=gender_model,
                                   emotion_model_path=emotion_model)
        self._recognizer = Recognizer(self._detector)
        self._swapper = Swapper(self._detector, model_path=swap_model, arcface_model_path=arcface_model)

        self._webcam = WebcamRecognizer(
            self._detector,
            self._recognizer,
            emotion_detector=self._emotion_detector,
            swapper=self._swapper,
        )

    # ---- Detection ----
    def detect(self, image, min_confidence=0.5):
        return self._detector.detect(image, min_confidence=min_confidence)

    # ---- Swapping (static images) ----
    def swap(self, source, target, output=None):
        return self._swapper.swap(source, target, output=output)

    # ---- Comparison ----
    def compare(self, image_a, image_b):
        return self._recognizer.compare(image_a, image_b)

    # ---- Recognition ----
    def register(self, name, image):
        self._recognizer.register(name, image)

    def recognize(self, image, threshold=65.0):
        return self._recognizer.recognize(image, threshold=threshold)

    # ---- Landmarks ----
    def landmarks(self, image):
        return self._landmarks.get(image)

    # ---- Analysis (age/gender/emotion) ----
    def analyze(self, image, include_emotion=True):
        return self._analyzer.analyze(image, include_emotion=include_emotion)

    # ---- Live webcam recognition (+ optional emotion overlay) ----
    def webcam_recognize(self, camera_index=0, threshold=65.0, show_emotion=False,
                           detect_scale=0.5, detect_every=3, emotion_every=6):
        self._webcam.run(
            camera_index=camera_index,
            threshold=threshold,
            show_emotion=show_emotion,
            detect_scale=detect_scale,
            detect_every=detect_every,
            emotion_every=emotion_every,
        )

    # ---- Live webcam face swap ----
    def webcam_swap(self, source_image, camera_index=0, swap_every=2):
        self._webcam.swap_live(source_image, camera_index=camera_index, swap_every=swap_every)