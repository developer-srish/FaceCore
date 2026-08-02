"""
Face swapping — runs the `inswapper_128.onnx` model directly via
onnxruntime. Includes its own 5-point similarity-transform alignment
step, implemented with plain OpenCV/numpy (see utils.align_face).

Source-face embeddings are computed with a real ArcFace ONNX model
(see arcface.py) -- this is required because inswapper_128 was trained
against ArcFace-style 512-d embeddings specifically. Using a different
recognition model's embeddings (e.g. dlib's 128-d vectors) produces a
swap that doesn't resemble the source face, since the embedding spaces
aren't related.
"""

import os
import numpy as np
import cv2
from .utils import load_image_rgb, to_bgr, to_rgb, save_image, align_face
from .arcface import ArcFaceEmbedder

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_SWAPPER_PATH = os.path.expanduser("~/.easyface/models/inswapper_128.onnx")


class Swapper:
    def __init__(self, detector, model_path=None, arcface_model_path=None):
        self._detector = detector
        self._model_path = model_path or DEFAULT_SWAPPER_PATH
        self._session = None
        self._emap = None
        self._embedder = ArcFaceEmbedder(model_path=arcface_model_path)

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
        from onnx import numpy_helper
        model_proto = onnx.load(self._model_path)
        # This model's export names the 512x512 embedding-projection matrix
        # 'buff2fs' (buffalo/ArcFace -> face-swap latent space) rather than
        # the 'emap' name used in some other builds/docs. Accept either name,
        # falling back to any (512, 512) 2-D initializer if neither matches,
        # since this matrix is applied directly by name in code (not wired
        # into the graph), which is also why onnxruntime logs it as unused.
        for init in model_proto.graph.initializer:
            if init.name in ("emap", "buff2fs"):
                self._emap = numpy_helper.to_array(init).astype(np.float32)
                break
        else:
            for init in model_proto.graph.initializer:
                arr = numpy_helper.to_array(init)
                if arr.ndim == 2 and arr.shape == (512, 512):
                    self._emap = arr.astype(np.float32)
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

        if source_face["landmarks"] is None:
            raise ValueError("Could not extract landmarks for source face alignment.")

        target_rgb = load_image_rgb(target)
        source_rgb = load_image_rgb(source)

        aligned_rgb, M = align_face(target_rgb, target_face["landmarks"], output_size=128)
        # inswapper expects RGB input (insightface's own code loads the aligned
        # crop as BGR via cv2 then does swapRB=True when building the blob --
        # net effect: RGB in). Feeding it BGR here caused the visible color
        # shift (e.g. skin/beard rendering green/off-colored) in the output.
        aligned_input = aligned_rgb.astype(np.float32) / 255.0
        blob = aligned_input.transpose(2, 0, 1)[np.newaxis, ...]

        emb_512 = self._embedder.embed(source_rgb, source_face["landmarks"])

        latent = self._get_source_latent(emb_512)

        input_names = [inp.name for inp in session.get_inputs()]
        feeds = {input_names[0]: blob.astype(np.float32), input_names[1]: latent}
        result = session.run(None, feeds)[0][0]

        # Model output is RGB (matches RGB input); no channel swap needed here.
        swapped_rgb = (result.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)

        inv_M = cv2.invertAffineTransform(M)
        h, w = target_rgb.shape[:2]
        warped_back = cv2.warpAffine(to_bgr(swapped_rgb), inv_M, (w, h), borderValue=0)

        # A hard-edged mask (all-ones, no feathering) produces a visible
        # rectangular seam at the paste boundary. Zero out a border strip
        # and blur it so the blend fades out smoothly instead of cutting
        # off sharply.
        mask = np.ones((128, 128), dtype=np.float32)
        border = 128 // 8
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=128 * 0.05)

        mask_warped = cv2.warpAffine(mask, inv_M, (w, h), borderValue=0)
        mask_warped = np.clip(mask_warped, 0.0, 1.0)[:, :, np.newaxis]

        target_bgr = to_bgr(target_rgb).astype(np.float32)
        composited = warped_back.astype(np.float32) * mask_warped + target_bgr * (1 - mask_warped)
        result_rgb = to_rgb(composited.astype(np.uint8))

        if output:
            save_image(result_rgb, output)

        return result_rgb