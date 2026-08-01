"""
Face swapping — runs the `inswapper_128.onnx` model directly via
onnxruntime. Reimplements the 5-point similarity-transform alignment
that insightface normally does internally, using pure OpenCV/numpy
(see utils.align_face).
"""

import os
import numpy as np
import cv2
from .utils import load_image_rgb, to_bgr, to_rgb, save_image, align_face

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_SWAPPER_PATH = os.path.expanduser("~/.easyface/models/inswapper_128.onnx")


class Swapper:
    def __init__(self, detector, model_path=None):
        self._detector = detector
        self._model_path = model_path or DEFAULT_SWAPPER_PATH
        self._session = None
        self._emap = None

    def _load_model(self):
        if self._session is not None:
            return self._session

        if ort is None:
            raise ImportError(
                "onnxruntime is required for face swapping. Install with: uv add onnxruntime"
            )

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Swap model not found at {self._model_path}. "
                "Download 'inswapper_128.onnx' and place it there, or pass "
                "model_path=... to Swapper()/Face(swap_model='...')"
            )

        self._session = ort.InferenceSession(
            self._model_path, providers=["CPUExecutionProvider"]
        )

        import onnx
        model_proto = onnx.load(self._model_path)
        for init in model_proto.graph.initializer:
            if init.name == "emap":
                self._emap = np.array(
                    onnx.numpy_helper.to_array(init), dtype=np.float32
                )
                break

        return self._session

    def _get_source_latent(self, source_embedding):
        """Project a recognition embedding into inswapper's latent space."""
        emb = np.asarray(source_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(emb)
        emb = emb / norm if norm != 0 else emb
        if self._emap is not None:
            latent = emb @ self._emap
        else:
            latent = emb
        latent = latent / np.linalg.norm(latent)
        return latent.astype(np.float32)

    def swap(self, source, target, output=None):
        """
        Swap the primary face from `source` onto the primary face in `target`.

        Returns the resulting RGB numpy array, and also saves to `output`
        if a path is given.
        """
        session = self._load_model()

        source_face = self._detector.largest_face(source)
        target_face = self._detector.largest_face(target)

        if source_face is None:
            raise ValueError("No face detected in source image.")
        if target_face is None:
            raise ValueError("No face detected in target image.")
        if target_face["landmarks"] is None:
            raise ValueError("Could not extract landmarks for target face alignment.")

        target_rgb = load_image_rgb(target)
        source_rgb = load_image_rgb(source)

        aligned_rgb, M = align_face(target_rgb, target_face["landmarks"], output_size=128)
        aligned_bgr = to_bgr(aligned_rgb).astype(np.float32) / 255.0
        blob = aligned_bgr.transpose(2, 0, 1)[np.newaxis, ...]

        from .recognition import Recognizer as _RecognizerClass
        _tmp_recognizer = _RecognizerClass(self._detector)
        raw_emb = _tmp_recognizer._embed(source_rgb, source_face["bbox"])

        emb_512 = np.zeros(512, dtype=np.float32)
        emb_512[: len(raw_emb)] = raw_emb

        latent = self._get_source_latent(emb_512)

        input_names = [inp.name for inp in session.get_inputs()]
        feeds = {input_names[0]: blob.astype(np.float32), input_names[1]: latent}
        result = session.run(None, feeds)[0][0]

        swapped_bgr = (result.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        swapped_rgb = to_rgb(swapped_bgr)

        inv_M = cv2.invertAffineTransform(M)
        h, w = target_rgb.shape[:2]
        warped_back = cv2.warpAffine(to_bgr(swapped_rgb), inv_M, (w, h), borderValue=0)
        mask = np.ones((128, 128), dtype=np.float32)
        mask_warped = cv2.warpAffine(mask, inv_M, (w, h), borderValue=0)
        mask_warped = mask_warped[:, :, np.newaxis]

        target_bgr = to_bgr(target_rgb).astype(np.float32)
        composited = warped_back.astype(np.float32) * mask_warped + target_bgr * (1 - mask_warped)
        result_rgb = to_rgb(composited.astype(np.uint8))

        if output:
            save_image(result_rgb, output)

        return result_rgb