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
import numpy as np
from .utils import to_bgr, to_rgb, euclidean_to_score


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
            score = euclidean_to_score(embedding, gal_embedding)
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
        last_results = []

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
                        x1, y1, x2, y2 = [v / detect_scale for v in f["bbox"]]

                        frame_rgb_full = to_rgb(frame_bgr)
                        try:
                            embedding = self._recognizer._embed(frame_rgb_full, (x1, y1, x2, y2))
                            name, score = self._best_match(embedding, threshold)
                            label = name if name == "Unknown" else f"{name} ({score:.0f}%)"
                        except Exception:
                            label = "Unknown"

                        entry = {"bbox": (int(x1), int(y1), int(x2), int(y2)),
                                 "label": label, "emotion": None}

                        if run_emotion and self._emotion is not None:
                            try:
                                emo = self._emotion.detect(frame_rgb_full, bbox=entry["bbox"])
                                if emo:
                                    entry["emotion"] = emo["emotion"]
                            except Exception:
                                pass

                        new_results.append(entry)

                    last_results = new_results

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

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        frame_count = 0
        last_output = None
        last_error = None

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_count += 1

                if frame_count % swap_every == 0:
                    frame_rgb = to_rgb(frame_bgr)
                    try:
                        swapped_rgb = self._swapper.swap(source_image, frame_rgb)
                        last_output = to_bgr(swapped_rgb)
                        last_error = None
                    except Exception as e:
                        # Surface the error instead of silently eating it --
                        # print once per distinct failure so it's visible,
                        # and fall back to the raw frame rather than a stale one.
                        msg = str(e)
                        if msg != last_error:
                            print(f"[webcam_swap] swap failed: {msg}")
                            last_error = msg

                display = last_output if (last_output is not None and last_error is None) else frame_bgr

                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()