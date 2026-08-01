"""
Age & gender analysis — reads attributes already produced by the
insightface detection model (genderage model bundled in buffalo_l).
Emotion is bolted on separately via EmotionDetector since insightface
doesn't ship an emotion model.
"""

from .emotion import EmotionDetector


class Analyzer:
    def __init__(self, detector):
        self._detector = detector
        self._emotion = EmotionDetector()

    def analyze(self, image, include_emotion=True):
        """
        Return age/gender (and optionally emotion) info for detected face(s).

        {
            "age": int,
            "gender": "Male" | "Female",
            "confidence": float,
            "emotion": "happy",          # only if include_emotion=True and fer succeeds
            "emotion_scores": {...}      # per-emotion breakdown
        }

        Returns None if no face is detected.
        Returns a list of these dicts if multiple faces are detected.
        """
        faces = self._detector.detect(image)
        if not faces:
            return None

        infos = []
        for f in faces:
            raw = f["_raw"]
            age = int(getattr(raw, "age", -1))
            gender_val = getattr(raw, "gender", None)
            gender = "Male" if gender_val == 1 else "Female" if gender_val == 0 else "Unknown"

            info = {
                "age": age,
                "gender": gender,
                "confidence": f["confidence"],
            }

            if include_emotion:
                try:
                    emo = self._emotion.detect(image, bbox=f["bbox"])
                    if emo:
                        info["emotion"] = emo["emotion"]
                        info["emotion_scores"] = emo["scores"]
                except ImportError as e:
                    info["emotion"] = None
                    info["emotion_error"] = str(e)

            infos.append(info)

        return infos[0] if len(infos) == 1 else infos