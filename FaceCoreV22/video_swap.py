"""
Face swap on video files: processes every frame through Swapper and
writes a new (silent) video. No audio muxing, no ffmpeg dependency.
"""

import cv2
from .utils import to_bgr, to_rgb


class VideoSwapper:
    def __init__(self, swapper):
        self._swapper = swapper

    def swap_video(self, source_image, target_video, output_path,
                    swap_every=1, show_progress=True):
        """
        Swap the primary face from `source_image` onto every frame of
        `target_video`, writing the result to `output_path`. Output has
        no audio track.

        swap_every: only run the (expensive) swap model every N frames;
        reuses the last swapped frame's result in between for speed.
        Set to 1 to swap every frame (best quality, slowest).
        """
        cap = cv2.VideoCapture(target_video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {target_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        last_output = None

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_count += 1

                if frame_count % swap_every == 0 or last_output is None:
                    frame_rgb = to_rgb(frame_bgr)
                    try:
                        swapped_rgb = self._swapper.swap(source_image, frame_rgb)
                        last_output = to_bgr(swapped_rgb)
                    except ValueError:
                        # no face detected in this frame -- keep original
                        last_output = frame_bgr

                writer.write(last_output)

                if show_progress and total_frames > 0 and frame_count % 30 == 0:
                    pct = frame_count / total_frames * 100
                    print(f"\rSwapping frames: {frame_count}/{total_frames} ({pct:.0f}%)", end="")

            if show_progress:
                print()
        finally:
            cap.release()
            writer.release()

        return output_path