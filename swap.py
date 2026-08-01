"""
Face swapping — wraps insightface's inswapper model.

Requires the inswapper_128.onnx model file. easyface will look for it
at ~/.insightface/models/inswapper_128.onnx by default (the standard
insightface cache location); download it separately since it is not
bundled with this package.
"""

import os
from .utils import load_image_rgb, to_bgr, to_rgb, save_image

try:
    import insightface
except ImportError:
    insightface = None


DEFAULT_SWAPPER_PATH = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")


class Swapper:
    def __init__(self, detector, model_path=None):
        self._detector = detector
        self._model_path = model_path or DEFAULT_SWAPPER_PATH
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        if insightface is None:
            raise ImportError("insightface is required for face swapping. Install with: pip install insightface")

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Swap model not found at {self._model_path}. "
                "Download 'inswapper_128.onnx' and place it there, or pass model_path=Face(swap_model='...')"
            )

        self._model = insightface.model_zoo.get_model(self._model_path)
        return self._model

    def swap(self, source, target, output=None):
        """
        Swap the primary face from `source` onto the primary face in `target`.

        Returns the resulting RGB numpy array, and also saves to `output`
        if a path is given.
        """
        model = self._load_model()

        source_face = self._detector.largest_face(source)
        target_face = self._detector.largest_face(target)

        if source_face is None:
            raise ValueError("No face detected in source image.")
        if target_face is None:
            raise ValueError("No face detected in target image.")

        target_rgb = load_image_rgb(target)
        target_bgr = to_bgr(target_rgb)

        result_bgr = model.get(
            target_bgr,
            target_face["_raw"],
            source_face["_raw"],
            paste_back=True,
        )

        result_rgb = to_rgb(result_bgr)

        if output:
            save_image(result_rgb, output)

        return result_rgb