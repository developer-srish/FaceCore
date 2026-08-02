"""
ArcFace embedding for face swapping.

inswapper_128.onnx was trained against 512-d ArcFace-style embeddings
(the same family of recognition model insightface uses internally --
commonly distributed as 'w600k_r50.onnx' / 'arcface_r100.onnx' /
similar). Feeding it embeddings from a *different* recognition model
(e.g. dlib's 128-d ResNet, even zero-padded to 512-d) does not work --
the two embedding spaces aren't related, so the swap output won't
resemble the source face at all.

This module runs a real ArcFace ONNX model directly through
onnxruntime, independent of dlib, specifically for swap.py's source
latent step. Recognizer (dlib-based) is unaffected and still used for
compare()/recognize().
"""

import os
import numpy as np
import cv2
from .utils import load_image_rgb, align_face

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_ARCFACE_MODEL = os.path.expanduser("~/.easyface/models/w600k_r50.onnx")


class ArcFaceEmbedder:
    def __init__(self, model_path=None):
        self._model_path = model_path or DEFAULT_ARCFACE_MODEL
        self._session = None

    def _load(self):
        if self._session is not None:
            return self._session

        if ort is None:
            raise ImportError(
                "onnxruntime is required for ArcFace embedding. "
                "Install with: uv add onnxruntime"
            )

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"ArcFace recognition model not found at {self._model_path}. "
                "This is required for face swapping (separate from the dlib "
                "model used for compare()/recognize()). Download an ArcFace "
                "ONNX model such as 'w600k_r50.onnx' (from the buffalo_l "
                "insightface model pack) and place it there, or pass "
                "arcface_model_path=... to Face()/Swapper()."
            )

        self._session = ort.InferenceSession(self._model_path, providers=["CPUExecutionProvider"])
        return self._session

    def _preprocess(self, aligned_rgb):
        # ArcFace/w600k_r50 expects RGB input (NOT BGR) -- normalized with
        # mean 127.5, std 127.5. Converting to BGR here was the bug: it
        # silently swapped the R/B channels, which still produces a
        # plausible-looking 512-d vector but points to the wrong region of
        # embedding space, so the swap output didn't resemble the source face.
        rgb = aligned_rgb.astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        return tensor

    def embed(self, image, landmarks_5pt):
        """
        Compute a 512-d ArcFace embedding for a face, given its 5-point
        landmarks (as returned by Detector.detect()). Aligns the face the
        same way inswapper's target alignment does before embedding.
        """
        session = self._load()
        rgb = load_image_rgb(image)
        aligned_rgb, _ = align_face(rgb, landmarks_5pt, output_size=112)
        tensor = self._preprocess(aligned_rgb)

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: tensor})[0][0]
        return np.asarray(output, dtype=np.float32)