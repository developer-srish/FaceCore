"""
Face detection wrapper around insightface's FaceAnalysis app.
Returns simple, easy-to-read dictionaries instead of raw model objects.
"""

from .utils import load_image_rgb, to_bgr


class Detector:
    def __init__(self, app):
        # app is a shared insightface.app.FaceAnalysis instance
        self._app = app

    def detect(self, image, min_confidence=0.5):
        """
        Detect faces in an image.

        Returns a list of dicts:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "landmarks": [[x, y], ...],   # 5-point landmarks
                "embedding": np.ndarray,      # 512-d face embedding
                "_raw": <insightface Face object>  # for internal reuse
            }
        """
        rgb = load_image_rgb(image)
        bgr = to_bgr(rgb)

        raw_faces = self._app.get(bgr)

        results = []
        for f in raw_faces:
            score = float(getattr(f, "det_score", 1.0))
            if score < min_confidence:
                continue

            results.append({
                "bbox": [float(v) for v in f.bbox],
                "confidence": round(score, 4),
                "landmarks": f.kps.tolist() if f.kps is not None else None,
                "embedding": f.normed_embedding if hasattr(f, "normed_embedding") else f.embedding,
                "_raw": f,
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