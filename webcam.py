"""
Real-time webcam features:
- recognize(): live face recognition with name overlay
- recognize(show_emotion=True): also overlays live emotion detection
- swap_live(): live face swap

Speed optimizations:
- Detection runs on a downscaled frame, boxes are scaled back up
- Full detection only runs every `detect_every` frames; in between,
  the last known boxes are reused (tracked, not re-detected)
- Emotion detection (heaviest per-face cost) runs even less often
"""

import cv2
from .utils import to_bgr, to_rgb, cosine_similarity


class WebcamRecognizer:
    def __init__(self, detector, recognizer, emotion_detector=None, swapper=None):
        self._detector = detector
        self._recognizer = recognizer
        self._emotion = emotion_detector
        self._swapper = swapper

    def _best_match(self, embedding, threshold):
        if not self._recognizer._gallery:
            return "Unknown", 0.0

        best_name, best_score = "Unknown", -1.0
        for name, gal_embedding in self._recognizer._gallery.items():
            score = cosine_similarity(embedding, gal_embedding)
            if score > best_score:
                best_name, best_score = name, score

        if best_score >= threshold:
            return best_name, best_score
        return "Unknown", best_score

    def run(self, camera_index=0, threshold=65.0, show_emotion=False,
             window_name="FaceCore - press q to quit",
             detect_scale=0.5, detect_every=3, emotion_every=6):
        """
        detect_scale : shrink frame before detection (0.5 = half size = ~4x faster detect)
        detect_every : run full detection every N frames; reuse boxes in between
        emotion_every: run emotion model every N frames (it's the slowest part)
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        frame_count = 0
        last_results = []  # cached: list of dicts with bbox, label

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_count += 1
                run_detect = (frame_count % detect_every == 0) or not last_results
                run_emotion = show_emotion and (frame_count % emotion_every == 0)

                if run_detect:
                    small_bgr = cv2.resize(frame_bgr, None, fx=detect_scale, fy=detect_scale)
                    small_rgb = to_rgb(small_bgr)
                    faces = self._detector.detect(small_rgb)

                    new_results = []
                    for f in faces:
                        # scale bbox back up to original frame size
                        x1, y1, x2, y2 = [v / detect_scale for v in f["bbox"]]
                        name, score = self._best_match(f["embedding"], threshold)
                        label = name if name == "Unknown" else f"{name} ({score:.0f}%)"

                        entry = {"bbox": (int(x1), int(y1), int(x2), int(y2)),
                                 "label": label, "emotion": None}

                        if run_emotion and self._emotion is not None:
                            try:
                                frame_rgb_full = to_rgb(frame_bgr)
                                emo = self._emotion.detect(frame_rgb_full, bbox=entry["bbox"])
                                if emo:
                                    entry["emotion"] = emo["emotion"]
                            except Exception:
                                pass

                        new_results.append(entry)

                    last_results = new_results

                # draw cached (or freshly updated) results every frame — cheap
                for entry in last_results:
                    x1, y1, x2, y2 = entry["bbox"]
                    label = entry["label"]
                    if entry["emotion"]:
                        label += f" | {entry['emotion']}"

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow(window_name, frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def swap_live(self, source_image, camera_index=0,
                   window_name="FaceCore - live swap - press q to quit",
                   swap_every=2):
        """
        swap_every: only run the (expensive) swap model every N frames;
        reuse the last swapped frame in between for smoother playback feel.
        """
        if self._swapper is None:
            raise ValueError("No swap model configured. Pass swap_model=... when creating Face().")

        source_face = self._detector.largest_face(source_image)
        if source_face is None:
            raise ValueError("No face detected in source image.")

        model = self._swapper._load_model()

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        frame_count = 0
        last_output = None

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_count += 1

                if frame_count % swap_every == 0:
                    frame_rgb = to_rgb(frame_bgr)
                    target_face = self._detector.largest_face(frame_rgb)

                    if target_face is not None:
                        try:
                            frame_bgr = model.get(
                                frame_bgr,
                                target_face["_raw"],
                                source_face["_raw"],
                                paste_back=True,
                            )
                        except Exception:
                            pass
                    last_output = frame_bgr
                elif last_output is not None:
                    frame_bgr = last_output

                cv2.imshow(window_name, frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
