# 🧠 FaceCore

<p align="center">
  <img src="assets/logo.png" alt="FaceCore Logo" width="300">
</p>

<h1 align="center">FaceCore</h1>

<p align="center">
<b>Fast • Accurate • AI-Powered Face Analysis & Face Swapping Library</b>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-success?style=for-the-badge)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange?style=for-the-badge)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Detection-yellow?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-success?style=for-the-badge)
![dlib](https://img.shields.io/badge/dlib-Face%20Recognition-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)

</p>

---

# 📖 About

**FaceCore** is a modern Python library for AI-powered face analysis.

It provides a simple, beginner-friendly API for:

- Face Detection
- Face Recognition
- Face Comparison
- Face Registration
- Facial Landmarks
- Age Detection
- Gender Detection
- Emotion Recognition
- Image Face Swapping
- Video Face Swapping
- Live Webcam Recognition
- Live Webcam Face Swapping

Unlike many face AI libraries, **FaceCore** hides the complexity of loading models, preprocessing images, and running inference. Everything can be done with just a few lines of Python.

Whether you're creating a security system, attendance software, AI camera application, or computer vision project, FaceCore offers an easy-to-use interface while still providing production-quality performance.

---

# ✨ Features

- ✅ Multi-face Detection
- ✅ Face Recognition
- ✅ Face Registration
- ✅ Face Verification
- ✅ Face Comparison
- ✅ Face Embeddings
- ✅ Facial Landmark Detection
- ✅ Age Prediction
- ✅ Gender Prediction
- ✅ Emotion Recognition
- ✅ Image Face Swapping
- ✅ Video Face Swapping
- ✅ Webcam Face Recognition
- ✅ Webcam Face Swapping
- ✅ Optional Emotion Overlay
- ✅ High-Speed ONNX Runtime
- ✅ MediaPipe Face Detection
- ✅ dlib Face Recognition
- ✅ Beginner Friendly API
- ✅ Cross Platform
- ✅ MIT Licensed

---

# ⚡ Requirements

- Python **3.12+**
- Windows, Linux or macOS
- FFmpeg (Required only for `swap_video()`)
- Webcam (Only required for live features)

---

# 📦 Installation

## Using pip

```bash
pip install FaceCoreV22
```

or

```bash
pip install mediapipe onnxruntime onnx opencv-python numpy pillow dlib-bin
```

---

## Using uv (Recommended)

```bash
uv add mediapipe onnxruntime onnx opencv-python numpy pillow dlib-bin
```

Then synchronize the environment

```bash
uv sync
```

---

# 🎞 Install FFmpeg

FFmpeg is only required for

- Video Face Swapping
- Preserving Audio

## Windows

```powershell
winget install ffmpeg
```

Restart the terminal afterwards.

Check installation

```bash
ffmpeg -version
```

---

## Linux

Ubuntu/Debian

```bash
sudo apt update
sudo apt install ffmpeg
```

Arch Linux

```bash
sudo pacman -S ffmpeg
```

Fedora

```bash
sudo dnf install ffmpeg
```

---

## macOS

```bash
brew install ffmpeg
```

Verify

```bash
ffmpeg -version
```

---

# 📁 Project Structure

```
FaceCore/
│
├── FaceCoreV22/
│   ├── detector.py
│   ├── recognition.py
│   ├── analysis.py
│   ├── landmarks.py
│   ├── swap.py
│   ├── video_swap.py
│   ├── webcam.py
│   └── core.py
│
├── assets/
│   └── logo.png
│
├── README.md
├── LICENSE
└── pyproject.toml
```

---

# 🤖 Model Directory

By default FaceCore searches for models inside

### Windows

```
C:\Users\<Username>\.easyface\models\
```

### Linux/macOS

```
~/.easyface/models/
```

You may also specify model paths manually while creating the `Face()` object.

---

# 🚀 Quick Start

```python
from FaceCoreV22 import Face

face = Face()
```

Or specify custom models

```python
from FaceCoreV22 import Face

face = Face(
    face_model="face_landmarker.task",
    age_model="age_googlenet.onnx",
    gender_model="gender_googlenet.onnx",
    emotion_model="emotion-ferplus-8.onnx",
    swap_model="inswapper_128.onnx"
)
```

Initialize the object only once and reuse it throughout your program.

---

# 📦 Dependencies

FaceCore is built using

- MediaPipe
- ONNX Runtime
- OpenCV
- dlib
- Pillow
- NumPy
- FFmpeg

These libraries are automatically used internally, allowing you to work with a clean, minimal API.

---

➡️ **Part 2 will include:**

- Complete Model Download Table
- Direct Download Links
- Folder Structure
- How to Extract `.bz2`
- Windows/Linux/macOS Model Locations
- Manual Model Loading
- Troubleshooting Missing Models
```
# 📥 Download Required Models

FaceCore does **not** automatically download AI models.

Download each model below and place it inside the default model directory.

## Default Model Directory

### Windows

```text
C:\Users\<YourUsername>\.easyface\models\
```

### Linux/macOS

```text
~/.easyface/models/
```

---

# 📦 Required Models

| Model | Required | Purpose | Download |
|-------|----------|---------|----------|
| face_landmarker.task | ✅ Yes | Face Detection & Landmarks | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task |
| dlib_face_recognition_resnet_model_v1.dat | ✅ Yes | Face Recognition | http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 |
| shape_predictor_5_face_landmarks.dat | ✅ Yes | Face Alignment | http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 |
| age_googlenet.onnx | Optional | Age Detection | https://huggingface.co/onnxmodelzoo/age_googlenet/resolve/main/age_googlenet.onnx |
| gender_googlenet.onnx | Optional | Gender Detection | https://huggingface.co/onnxmodelzoo/gender_googlenet/resolve/main/gender_googlenet.onnx |
| emotion-ferplus-8.onnx | Optional | Emotion Recognition | https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx |
| inswapper_128.onnx | Optional | Face Swapping | https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx |

---

# 📂 Folder Structure

After downloading all models your folder should look like this:

```text
.easyface/
└── models/
    ├── face_landmarker.task
    ├── dlib_face_recognition_resnet_model_v1.dat
    ├── shape_predictor_5_face_landmarks.dat
    ├── age_googlenet.onnx
    ├── gender_googlenet.onnx
    ├── emotion-ferplus-8.onnx
    └── inswapper_128.onnx
```

---

# 📥 Extracting dlib Models

The dlib models are downloaded as **.bz2** files.

## Windows

Extract them using

- 7-Zip
- WinRAR

After extracting you should get

```text
dlib_face_recognition_resnet_model_v1.dat
shape_predictor_5_face_landmarks.dat
```

Copy both files into

```text
C:\Users\<Username>\.easyface\models\
```

---

## Linux

Install bzip2

```bash
sudo apt install bzip2
```

Extract

```bash
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 shape_predictor_5_face_landmarks.dat.bz2
```

Move

```bash
mkdir -p ~/.easyface/models

mv dlib_face_recognition_resnet_model_v1.dat ~/.easyface/models/

mv shape_predictor_5_face_landmarks.dat ~/.easyface/models/
```

---

## macOS

```bash
brew install bzip2

bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2

bunzip2 shape_predictor_5_face_landmarks.dat.bz2
```

Move

```bash
mkdir -p ~/.easyface/models

mv dlib_face_recognition_resnet_model_v1.dat ~/.easyface/models/

mv shape_predictor_5_face_landmarks.dat ~/.easyface/models/
```

---

# 🖥 Manual Model Loading

Instead of using the default folder, you can manually specify model paths.

```python
from FaceCoreV22 import Face

face = Face(
    face_model="models/face_landmarker.task",
    age_model="models/age_googlenet.onnx",
    gender_model="models/gender_googlenet.onnx",
    emotion_model="models/emotion-ferplus-8.onnx",
    swap_model="models/inswapper_128.onnx"
)
```

You only need to provide the models you intend to use.

Example:

```python
from FaceCoreV22 import Face

face = Face(
    swap_model="models/inswapper_128.onnx"
)
```

---

# 📁 Model Usage

| Feature | Required Models |
|----------|-----------------|
| Face Detection | face_landmarker.task |
| Face Landmarks | face_landmarker.task |
| Face Recognition | face_landmarker.task + dlib_face_recognition_resnet_model_v1.dat + shape_predictor_5_face_landmarks.dat |
| Face Comparison | face_landmarker.task + dlib_face_recognition_resnet_model_v1.dat + shape_predictor_5_face_landmarks.dat |
| Face Registration | face_landmarker.task + dlib_face_recognition_resnet_model_v1.dat + shape_predictor_5_face_landmarks.dat |
| Age Detection | age_googlenet.onnx |
| Gender Detection | gender_googlenet.onnx |
| Emotion Detection | emotion-ferplus-8.onnx |
| Image Face Swap | inswapper_128.onnx |
| Video Face Swap | inswapper_128.onnx |
| Webcam Recognition | face_landmarker.task + dlib_face_recognition_resnet_model_v1.dat + shape_predictor_5_face_landmarks.dat |
| Webcam Face Swap | inswapper_128.onnx |

---

# ⚠️ Common Model Errors

## Recognition model not found

```
Recognition model not found
```

Download

```
dlib_face_recognition_resnet_model_v1.dat
```

and place it inside

```text
~/.easyface/models/
```

---

## Shape predictor not found

```
Shape predictor not found
```

Download

```
shape_predictor_5_face_landmarks.dat
```

and place it inside

```text
~/.easyface/models/
```

---

## Emotion model not found

```
emotion-ferplus-8.onnx not found
```

Download

```
emotion-ferplus-8.onnx
```

from Hugging Face.

---

## Age/Gender model not found

Download

- age_googlenet.onnx
- gender_googlenet.onnx

from Hugging Face.

---

## Face swap model not found

Download

```
inswapper_128.onnx
```

and place it in the models folder.

---

➡️ **Part 3 includes complete API documentation and examples for every FaceCore function:**

- detect()
- compare()
- register()
- recognize()
- landmarks()
- analyze()
- swap()
- swap_video()
- webcam_recognize()
- webcam_swap()
  # 📚 FaceCore API Documentation

Once you have initialized FaceCore,

```python
from FaceCoreV22 import Face

face = Face()
```

you can access every feature through the same object.

---

# 🔍 Face Detection

Detect every face inside an image.

## Syntax

```python
faces = face.detect(image_path)
```

## Example

```python
from FaceCoreV22 import Face

face = Face()

faces = face.detect("group.jpg")

print(faces)
```

## Returns

```python
[
    {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.99,
        "landmarks": [...]
    }
]
```

---

# 🆚 Face Comparison

Compare two face images.

Returns a similarity score between **0–100**.

## Syntax

```python
score = face.compare(image1, image2)
```

## Example

```python
score = face.compare(
    "me1.jpg",
    "me2.jpg"
)

print(score)
```

Output

```
96.74
```

Higher score = More similar.

---

# 👤 Register Face

Register a person's face.

Registered faces are stored in memory and can later be recognized.

## Syntax

```python
face.register(name, image)
```

## Example

```python
face.register(
    "Srish",
    "srish.jpg"
)
```

Register multiple people

```python
face.register("Alex","alex.jpg")
face.register("John","john.jpg")
face.register("Emma","emma.jpg")
```

---

# 🪪 Face Recognition

Recognize a registered person.

## Syntax

```python
name = face.recognize(image)
```

## Example

```python
name = face.recognize(
    "person.jpg"
)

print(name)
```

Output

```
Srish
```

You can also specify a threshold.

```python
name = face.recognize(
    "person.jpg",
    threshold=65
)
```

---

# 📍 Face Landmarks

Extract facial landmarks.

## Syntax

```python
points = face.landmarks(image)
```

## Example

```python
points = face.landmarks(
    "person.jpg"
)

print(points)
```

Returns

```python
[
    (x,y),
    (x,y),
    (x,y),
    (x,y),
    (x,y)
]
```

---

# 🎂 Age & Gender Detection

Estimate a person's age and gender.

## Syntax

```python
info = face.analyze(image)
```

## Example

```python
info = face.analyze(
    "person.jpg"
)

print(info)
```

Output

```python
{
    "age":"(25-32)",
    "gender":"Male",
    "confidence":0.99
}
```

---

# 😀 Emotion Detection

Requires

```
emotion-ferplus-8.onnx
```

## Syntax

```python
info = face.analyze(
    image,
    include_emotion=True
)
```

## Example

```python
info = face.analyze(
    "person.jpg",
    include_emotion=True
)

print(info)
```

Output

```python
{
    "age":"(25-32)",
    "gender":"Male",
    "emotion":"Happy",
    "confidence":0.98
}
```

---

# 🖼 Image Face Swap

Swap one face onto another image.

## Syntax

```python
face.swap(
    source,
    target,
    output
)
```

## Example

```python
face.swap(
    source="me.jpg",
    target="friend.jpg",
    output="result.jpg"
)
```

Output

```
result.jpg
```

---

# 🎬 Video Face Swap

Swap faces in a video.

Audio is automatically preserved using FFmpeg.

## Syntax

```python
face.swap_video(
    source_image,
    target_video,
    output_path
)
```

## Example

```python
face.swap_video(
    source_image="me.jpg",
    target_video="video.mp4",
    output_path="output.mp4"
)
```

Swap every second frame

```python
face.swap_video(
    source_image="me.jpg",
    target_video="video.mp4",
    output_path="output.mp4",
    swap_every=2
)
```

Swap every third frame

```python
face.swap_video(
    source_image="me.jpg",
    target_video="video.mp4",
    output_path="output.mp4",
    swap_every=3
)
```

---

# 📷 Webcam Recognition

Run real-time face recognition.

## Syntax

```python
face.webcam_recognize()
```

## Example

```python
face.webcam_recognize()
```

With emotion detection

```python
face.webcam_recognize(
    show_emotion=True
)
```

Without emotion detection

```python
face.webcam_recognize(
    show_emotion=False
)
```

---

# 🎭 Webcam Face Swap

Swap your face in real-time.

## Syntax

```python
face.webcam_swap(
    source_image
)
```

## Example

```python
face.webcam_swap(
    source_image="me.jpg"
)
```

Process every third frame

```python
face.webcam_swap(
    source_image="me.jpg",
    swap_every=3
)
```

---

# 📊 Feature Summary

| Method | Description |
|---------|-------------|
| `detect()` | Detect all faces in an image |
| `compare()` | Compare two faces |
| `register()` | Register a person's face |
| `recognize()` | Recognize registered faces |
| `landmarks()` | Extract facial landmarks |
| `analyze()` | Detect age, gender and emotion |
| `swap()` | Swap faces in an image |
| `swap_video()` | Swap faces in a video |
| `webcam_recognize()` | Live face recognition |
| `webcam_swap()` | Live face swapping |

---

# 💡 Tips

- Create the `Face()` object only once.
- Reuse the same object throughout your application.
- Store all AI models in the default `.easyface/models` directory.
- Use `swap_every=2` or `swap_every=3` for faster video processing.
- Ensure FFmpeg is installed before using `swap_video()`.
- For best recognition accuracy, use clear, front-facing images with good lighting.

---

➡️ **Part 4 includes:**

- Troubleshooting
- Frequently Asked Questions (FAQ)
- Performance Tips
- Roadmap
- Contributing
- License
- Author
- Credits
- Support
- Changelog
- Final footer
  # ⚠️ Troubleshooting

If you encounter any issues while using FaceCore, check the solutions below.

---

## ModuleNotFoundError

Example

```text
ModuleNotFoundError: No module named 'onnxruntime'
```

Install the required packages

```bash
pip install mediapipe onnxruntime onnx opencv-python numpy pillow dlib-bin
```

or

```bash
uv add mediapipe onnxruntime onnx opencv-python numpy pillow dlib-bin
```

---

## Recognition Model Not Found

```text
Recognition model not found
```

Download

```
dlib_face_recognition_resnet_model_v1.dat
```

Place it inside

```
~/.easyface/models/
```

---

## Shape Predictor Not Found

```text
Shape predictor not found
```

Download

```
shape_predictor_5_face_landmarks.dat
```

Place it inside

```
~/.easyface/models/
```

---

## Face Landmarker Not Found

```text
face_landmarker.task not found
```

Download

```
face_landmarker.task
```

Place it inside

```
~/.easyface/models/
```

---

## Emotion Model Missing

```text
emotion-ferplus-8.onnx not found
```

Download

```
emotion-ferplus-8.onnx
```

Place it inside

```
~/.easyface/models/
```

---

## Face Swap Model Missing

```text
inswapper_128.onnx not found
```

Download

```
inswapper_128.onnx
```

Place it inside

```
~/.easyface/models/
```

---

## No Face Detected

Possible reasons

- Poor lighting
- Face too small
- Face rotated
- Blurry image
- Face covered

---

## Webcam Not Opening

Check

- Webcam permissions
- Webcam drivers
- Another application isn't already using the camera

---

## Video Swap Doesn't Preserve Audio

Install FFmpeg

```bash
ffmpeg -version
```

If not installed

Windows

```powershell
winget install ffmpeg
```

Ubuntu

```bash
sudo apt install ffmpeg
```

---

## Slow Processing

For faster processing

```python
swap_every=2
```

or

```python
swap_every=3
```

instead of

```python
swap_every=1
```

---

# ❓ Frequently Asked Questions

### Does FaceCore require a GPU?

No.

FaceCore works on CPU.

ONNX Runtime automatically accelerates inference where possible.

---

### Does FaceCore support NVIDIA CUDA?

Currently the default build uses CPU.

CUDA support may be added in future releases.

---

### Can FaceCore detect multiple faces?

Yes.

All faces inside an image are detected.

---

### Can I register multiple people?

Yes.

```python
face.register("Alex","alex.jpg")
face.register("John","john.jpg")
face.register("Emma","emma.jpg")
```

---

### Does registration persist after restarting Python?

No.

Currently the face database is stored in memory.

Persistent storage is planned for a future release.

---

### Which image formats are supported?

- JPG
- JPEG
- PNG
- BMP
- WEBP

---

### Which video formats are supported?

- MP4
- AVI
- MOV
- MKV
- WMV

---

### Can I use FaceCore commercially?

Yes.

FaceCore is released under the MIT License.

Always verify that the AI models you download are licensed for your intended use.

---

# 🚀 Performance Tips

For best results

✅ Use clear images

✅ Good lighting

✅ Front-facing faces

✅ High-resolution images

✅ Keep models on SSD storage

✅ Reuse the same `Face()` object

Example

```python
face = Face()

face.detect(...)

face.compare(...)

face.swap(...)
```

Don't recreate the object repeatedly.

---

# 🛣 Roadmap

Upcoming features

- GPU Acceleration
- Batch Face Processing
- Face Enhancement
- Face Restoration
- Face Tracking
- Multiple Face Swapping
- Face Blur
- Face Mask Detection
- Face Quality Scoring
- Emotion Timeline
- Persistent Face Database
- GUI Application
- Command Line Interface
- Docker Support
- Mobile Support
- REST API
- Live Video Streaming
- Better Face Matching
- Performance Improvements

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the **MIT License**.

You are free to

- Use
- Modify
- Distribute
- Share

Please include the original license when redistributing.

---

# ❤️ Support

If you like FaceCore,

⭐ Star the repository

🐛 Report bugs

💡 Suggest new features

🤝 Contribute to the project

---

# 👨‍💻 Author

## Srish Ghosh

Python Developer

Open Source Enthusiast

GitHub

```
https://github.com/developer-srish
```

---

# 🙏 Credits

FaceCore is built using these amazing open-source projects.

- Python
- MediaPipe
- OpenCV
- ONNX Runtime
- dlib
- NumPy
- Pillow
- FFmpeg

Special thanks to the developers and maintainers of these projects.

---

# 📈 Changelog

## Version 2.2.0

### Added

- Face Detection
- Face Recognition
- Face Registration
- Face Comparison
- Face Landmarks
- Age Detection
- Gender Detection
- Emotion Detection
- Image Face Swapping
- Video Face Swapping
- Webcam Recognition
- Webcam Face Swapping

### Improved

- Faster ONNX Runtime inference
- Better detection accuracy
- Cleaner API
- Simpler installation
- Better documentation

### Fixed

- Windows compatibility
- Model loading improvements
- FFmpeg integration
- Recognition performance

---

# 🌟 Star History

If FaceCore helps you, consider giving the repository a ⭐ on GitHub.

It helps others discover the project and motivates future development.

---

<p align="center">

## ⭐ If you like FaceCore, don't forget to star the repository!

Made with ❤️ by **Srish Ghosh**

</p>
