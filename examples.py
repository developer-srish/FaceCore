"""
Generates simple placeholder "face" sample images using Pillow only,
so the package has runnable examples without shipping real photos.
These are stylized cartoon faces — good for testing detect()/landmarks()
pipelines, not for realistic swap/recognition demos.
"""

import os
import random
from PIL import Image, ImageDraw

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _draw_face(draw, cx, cy, size, skin_color):
    r = size // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=skin_color, outline=(60, 40, 30))

    eye_r = max(3, size // 12)
    eye_dx = r // 2
    eye_dy = -r // 5
    draw.ellipse([cx - eye_dx - eye_r, cy + eye_dy - eye_r,
                  cx - eye_dx + eye_r, cy + eye_dy + eye_r], fill="white", outline="black")
    draw.ellipse([cx + eye_dx - eye_r, cy + eye_dy - eye_r,
                  cx + eye_dx + eye_r, cy + eye_dy + eye_r], fill="white", outline="black")

    pupil_r = eye_r // 2
    draw.ellipse([cx - eye_dx - pupil_r, cy + eye_dy - pupil_r,
                  cx - eye_dx + pupil_r, cy + eye_dy + pupil_r], fill="black")
    draw.ellipse([cx + eye_dx - pupil_r, cy + eye_dy - pupil_r,
                  cx + eye_dx + pupil_r, cy + eye_dy + pupil_r], fill="black")

    nose_len = size // 6
    draw.line([cx, cy, cx, cy + nose_len], fill=(120, 80, 60), width=2)

    mouth_w = r // 1.2
    mouth_y = cy + r // 2
    draw.arc([cx - mouth_w, mouth_y - 15, cx + mouth_w, mouth_y + 15], start=20, end=160, fill="black", width=3)


def generate_sample(path, width=400, height=400, num_faces=1, seed=None):
    """Create one sample image with `num_faces` simple cartoon faces on it."""
    if seed is not None:
        random.seed(seed)

    img = Image.new("RGB", (width, height), color=(230, 230, 235))
    draw = ImageDraw.Draw(img)

    skin_tones = [(255, 224, 189), (241, 194, 125), (198, 134, 66), (141, 85, 36)]

    for _ in range(num_faces):
        size = random.randint(min(width, height) // 4, min(width, height) // 2)
        cx = random.randint(size, width - size) if width > 2 * size else width // 2
        cy = random.randint(size, height - size) if height > 2 * size else height // 2
        skin = random.choice(skin_tones)
        _draw_face(draw, cx, cy, size, skin)

    img.save(path)
    return path


def generate_all_samples():
    """Regenerate the three bundled sample images used in demos/tests."""
    os.makedirs(DATA_DIR, exist_ok=True)

    generate_sample(os.path.join(DATA_DIR, "sample1.jpg"), num_faces=1, seed=1)
    generate_sample(os.path.join(DATA_DIR, "sample2.jpg"), num_faces=1, seed=2)
    generate_sample(os.path.join(DATA_DIR, "sample3.jpg"), num_faces=3, seed=3)  # group photo


def sample_path(name):
    """Get the full path to a bundled sample image, e.g. sample_path('sample1.jpg')."""
    return os.path.join(DATA_DIR, name)


if __name__ == "__main__":
    generate_all_samples()
    print(f"Sample images generated in: {DATA_DIR}")