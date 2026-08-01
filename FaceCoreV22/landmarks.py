"""
Facial landmark extraction — thin convenience layer on top of Detector.
"""


class Landmarks:
    def __init__(self, detector):
        self._detector = detector

    def get(self, image):
        """
        Return landmark points for all faces found in the image.

        If a single face is found, returns just that face's points (list of [x, y]).
        If multiple faces are found, returns a list of point-lists, one per face.
        If no faces are found, returns None.
        """
        faces = self._detector.detect(image)
        if not faces:
            return None

        points = [f["landmarks"] for f in faces]
        return points[0] if len(points) == 1 else points