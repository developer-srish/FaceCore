"""
easyface/core.py

Face — the single friendly entry point that ties every module together.
"""

from .detector import Detector
from .landmarks import Landmarks
from .analysis import Analyzer
from .recognition import Recognizer
from .swap import Swapper
from .emotion import EmotionDetector
from .webcam import WebcamRecognizer

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    insightface = None
    FaceAnalysis = None


class Face:
    def __init__(self, model_name="buffalo_l", swap_model=None, ctx_id=0, providers=None):
        """
        model_name : insightface model pack to use for detection/recognition/analysis
        swap_model : optional path to an inswapper .onnx model (needed for swap features)
        ctx_id     : device id (0 = first GPU if available, -1 = force CPU)
        providers  : optional onnxruntime execution providers list
        """
        if FaceAnalysis is None:
            raise ImportError(
                "insightface is required. Install with: pip install insightface onnxruntime"
            )

        self._app = FaceAnalysis(name=model_name, providers=providers)
        self._app.prepare(ctx_id=ctx_id)

        # core building blocks
        self._detector = Detector(self._app)
        self._landmarks = Landmarks(self._detector)
        self._emotion_detector = EmotionDetector()
        self._analyzer = Analyzer(self._detector)
        self._recognizer = Recognizer(self._detector)
        self._swapper = Swapper(self._detector, model_path=swap_model)

        # webcam ties detector + recognizer + emotion + swapper together
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
        """
        Open a live webcam window with face recognition overlay.

        detect_scale : shrink frame before detection (0.5 = half size, faster)
        detect_every : run full detection every N frames; reuse boxes in between
        emotion_every: run emotion model every N frames (heaviest per-face cost)
        """
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
        """
        Open a live webcam window with the detected face swapped for
        the face in `source_image`.

        swap_every: only run the (expensive) swap model every N frames;
        reuses the last swapped frame in between for smoother playback.
        """
        self._webcam.swap_live(source_image, camera_index=camera_index, swap_every=swap_every)