# 🧠 FaceCore

### A Simple, Powerful Python Face Analysis Toolkit

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-success?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📖 About

**FaceCore** is a lightweight, beginner-friendly Python module for face detection, comparison, recognition, landmark extraction, and demographic/emotion analysis — all through a clean, minimal API.

It's built for developers who want production-grade face analysis without wrestling with low-level model loading, preprocessing, or ONNX plumbing.

---

## ✨ Features

- ✅ Multi-face detection with bounding boxes, confidence scores, landmarks, and embeddings
- ✅ Face similarity comparison (0–100 score)
- ✅ Face registration and recognition (identity matching)
- ✅ 2D facial landmark extraction
- ✅ Age and gender estimation
- ✅ Emotion detection (via ONNX emotion model)
- ✅ Real-time webcam recognition with optional emotion overlay
- ✅ Simple, readable API — one import, one object

---

## 📋 Requirements

- Python 3.9+
- pip
- A webcam (only required for live recognition features)

---

## 📦 Installation

```bash
pip install facecore
```

Or install from source:

```bash
git clone https://github.com/developer-srish/facecore.git
cd facecore
pip install -r requirements.txt
```

---

## 🤖 Model Setup

FaceCore relies on ONNX models for inference. At minimum, you'll need a face detection/recognition model. Emotion detection requires an additional model.

| Feature | Required Model |
|---|---|
| Detection / Recognition / Landmarks / Age-Gender | Core face model (bundled or configurable path) |
| Emotion Detection | `emotion-ferplus-8.onnx` |

Place your model files in a local `models/` directory and reference them when initializing `Face`.

---

## 🚀 Quick Start

```python
from facecore import Face

face = Face(
    model_path=r"C:\models\face_core.onnx",
)
```

> ℹ️ Initialize once and reuse the `face` object across all calls below.

---

## 🔍 Detection

Detect all faces in an image, returning bounding boxes, confidence scores, landmarks, and embeddings for each face found.

```python
faces = face.detect("group.jpg")
```

---

## 🆚 Comparison

Compare two face images and get a similarity score from 0–100.

```python
score = face.compare("me1.jpg", "me2.jpg")
```

---

## 🪪 Recognition

Register a known face, then recognize it in future images.

```python
face.register("srish", "srish.jpg")
name = face.recognize("person.jpg")
```

---

## 📍 Landmarks

Extract facial landmark points (eyes, nose, mouth, jawline, etc.).

```python
points = face.landmarks("person.jpg")
```

---

## 🎂 Age & Gender Analysis

```python
info = face.analyze("person.jpg", include_emotion=False)
```

---

## 😀 Emotion Detection

Requires the `emotion-ferplus-8.onnx` model.

```python
info = face.analyze("person.jpg", include_emotion=True)
```

---

## 📷 Live Webcam Recognition

Run real-time face recognition through your webcam, with an optional live emotion overlay.

```python
# With emotion overlay
face.webcam_recognize(show_emotion=True)

# Without emotion overlay
face.webcam_recognize(show_emotion=False)
```

---
# Static Face Swap

Swap one face into another image.

```python
face.swap(
    source="me.jpg",
    target="friend.jpg",
    output="output.jpg"
)
```

---

# Live Webcam Face Swap

Swap your face live using a webcam.

```python
face.webcam_swap(
    source_image="me.jpg",
    swap_every=3
)
```

---
# Download Required Models

EasyFace uses pretrained ONNX models for advanced features.

| Model | Purpose | Required | Hugging Face |
|--------|----------|----------|--------------|
| `buffalo_l` | Face Detection, Recognition, Landmarks | ✅ Yes | Automatically downloaded by InsightFace |
| `inswapper_128.onnx` | Face Swapping | ✅ Yes | https://huggingface.co/ezioruan/inswapper_128.onnx |
| `emotion-ferplus-8.onnx` | Emotion Detection | Optional | https://huggingface.co/onnxmodelzoo/emotion-ferplus-8 |

## 🛠️ Technologies Used

- 🐍 Python
- ⚡ ONNX Runtime
- 👁️ OpenCV
- 🔢 NumPy
- 🧠 Deep Learning / Computer Vision

---

## ⚠️ Troubleshooting

**ModuleNotFoundError**
```bash
pip install numpy opencv-python onnxruntime
```

**No Face Found**
- Ensure the face is clearly visible and well-lit
- Avoid heavy blur or extreme angles

**Emotion Detection Not Working**
- Confirm `emotion-ferplus-8.onnx` is present and its path is correctly referenced
- Only used when `include_emotion=True`

**CUDAExecutionProvider Warning**
```text
Specified provider 'CUDAExecutionProvider' is not available
```
This just means inference is running on CPU instead of GPU — the module will still function correctly.

---

## 📚 What You'll Learn

- Face detection & embeddings
- Face verification and 1:1 comparison
- Identity registration & recognition pipelines
- Facial landmark extraction
- Age, gender, and emotion inference
- Real-time webcam-based computer vision

---

## 🤝 Contributing

Contributions are welcome!

```bash
git checkout -b feature-name
git commit -m "Add new feature"
git push origin feature-name
```

Then open a Pull Request.

---

## 📜 License

Licensed under the **MIT License** — free to use, modify, and share.

---

## 👨‍💻 Author

**Srish Ghosh**
Python Developer • Open Source Enthusiast

GitHub: [github.com/developer-srish](https://github.com/developer-srish)

---

## 🌟 Roadmap

- 🎥 Video-based batch analysis
- 🖥️ GUI wrapper
- ⚡ GPU acceleration
- 📁 Batch image processing
- 🧠 Face enhancement pre-processing

---

<p align="center">Made with ❤️ in Python by <strong>Srish Ghosh</strong></p>
