"""
Face comparison & simple named-person recognition.

Recognition works against a small in-memory "gallery" that you can
register faces into with Face.register(name, image). This keeps the
library self-contained (no external DB) while still giving you
name -> face matching.
"""

from .utils import cosine_similarity


class Recognizer:
    def __init__(self, detector):
        self._detector = detector
        self._gallery = {}  # name -> embedding

    def register(self, name, image):
        """Add a named person to the recognition gallery."""
        face = self._detector.largest_face(image)
        if face is None:
            raise ValueError(f"No face found in image for '{name}'")
        self._gallery[name] = face["embedding"]

    def compare(self, image_a, image_b):
        """Return a 0-100 similarity score between the primary faces in two images."""
        face_a = self._detector.largest_face(image_a)
        face_b = self._detector.largest_face(image_b)

        if face_a is None or face_b is None:
            raise ValueError("Could not detect a face in one or both images.")

        return cosine_similarity(face_a["embedding"], face_b["embedding"])

    def recognize(self, image, threshold=65.0):
        """
        Compare the primary face in `image` against everyone in the gallery.
        Returns the best-matching name, or "Unknown" if nothing clears the threshold.
        """
        if not self._gallery:
            return "Unknown"

        face = self._detector.largest_face(image)
        if face is None:
            return "Unknown"

        best_name, best_score = "Unknown", -1.0
        for name, embedding in self._gallery.items():
            score = cosine_similarity(face["embedding"], embedding)
            if score > best_score:
                best_name, best_score = name, score

        return best_name if best_score >= threshold else "Unknown"