# 🧠 FaceCore
 <p>Website Link https://developer-srish.github.io/Facecore.com/</p>
 <p>Drive Link for Models if you can't download it from website:</p>

<p align="center">
  <img src="assets/logo.png" width="260" alt="FaceCore Logo">
</p>

<h1 align="center">FaceCore</h1>

<p align="center">
<b>Simple • Fast • AI-Powered Face Analysis Library for Python</b>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-success?style=for-the-badge)
![InsightFace](https://img.shields.io/badge/InsightFace-AI-blue?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)

</p>

---

# 📖 About

**FaceCore** is a modern Python library that brings multiple AI-powered face processing tools together behind a single, easy-to-use API.

Instead of working with several different libraries for detection, recognition, landmarks, analysis, emotion detection, and face swapping, FaceCore provides one `Face` class that initializes everything for you.

Whether you're creating an attendance system, security application, AI camera, computer vision project, or simply experimenting with facial AI, FaceCore lets you build it with only a few lines of Python.

---

# ✨ Features

## 👁 Face Detection

Detect one or multiple faces in an image with accurate bounding boxes.

- Multiple face detection
- Fast inference
- Confidence scores
- Face locations

---

## 👤 Face Recognition

Recognize previously registered people using facial embeddings.

- Face registration
- Face matching
- Adjustable recognition threshold
- Multiple identities

---

## 🆚 Face Comparison

Compare two face images and receive a similarity score.

Perfect for

- Identity verification
- Duplicate detection
- Authentication systems

---

## 📍 Facial Landmarks

Extract important facial landmarks including

- Eyes
- Nose
- Mouth
- Face outline

Useful for

- Face alignment
- Image processing
- Face swapping

---

## 🎂 Face Analysis

Analyze a detected face and estimate

- Age
- Gender

using deep learning models.

---

## 😀 Emotion Recognition

Optionally detect facial emotions including

- Happy
- Sad
- Angry
- Fear
- Surprise
- Neutral

---

## 🎭 Face Swapping

Swap one person's face onto another image using an ONNX face swap model.

Designed for

- Image editing
- AI demonstrations
- Research projects

---

## 📷 Webcam Recognition

Run live face recognition directly from your webcam.

Supports

- Real-time recognition
- Live detection
- Multiple faces

---

# ⭐ Why FaceCore?

- ✅ Single `Face()` class
- ✅ Beginner Friendly API
- ✅ Modular Design
- ✅ Fast AI Inference
- ✅ InsightFace Powered
- ✅ ONNX Runtime Support
- ✅ Cross Platform
- ✅ MIT Licensed

---

# ⚡ Requirements

| Requirement | Version |
|------------|---------|
| Python | 3.10+ |
| InsightFace | Latest |
| OpenCV | Latest |
| ONNX Runtime | Latest |
| Pillow | Latest |
| NumPy | Latest |

---

# 📦 Installation

Install FaceCore dependencies

```bash
pip install insightface onnxruntime opencv-python pillow numpy
```

or install from the requirements file

```bash
pip install -r requirements/requirements.txt
```

---

# 📁 Project Structure

```text
FaceCore/
│
├── __init__.py
├── core.py
├── detector.py
├── recognition.py
├── landmarks.py
├── analysis.py
├── emotion.py
├── swap.py
├── webcam.py
├── utils.py
├── examples.py
│
├── assets/
│   └── logo.png
│
├── requirements/
│   └── requirements.txt
│
├── LICENSE
└── README.md
```

---

# 🚀 Quick Start

```python
from FaceCore import Face

face = Face()
```

Using a custom face swap model

```python
from FaceCore import Face

face = Face(
    swap_model="inswapper_128.onnx"
)
```

Using a different InsightFace model pack

```python
from FaceCore import Face

face = Face(
    model_name="buffalo_l"
)
```

Initialize the `Face()` object only once and reuse it throughout your application for the best performance.

---

# 🏗 Architecture

The `Face` class combines multiple independent modules into a single interface.

```
                Face
                  │
 ┌────────────────┼────────────────┐
 │                │                │
Detector     Recognition      Analysis
 │                │                │
Landmarks     Emotion        Face Swap
                  │
             Webcam Support
```

This modular architecture keeps the library simple for beginners while remaining flexible for advanced applications.

---

# 📦 Built With

FaceCore is powered by several open-source projects.

- InsightFace
- ONNX Runtime
- OpenCV
- Pillow
- NumPy

These libraries work behind the scenes, allowing developers to interact with a clean and minimal API.

---

# 📚 What's Next?

**Part 2** includes

- 📥 Required AI Models
- 🌐 Official Download Links
- 📂 Default Model Locations
- 🖥 Windows Setup
- 🐧 Linux Setup
- 🍎 macOS Setup
- 📦 Manual Model Loading
- 🧩 InsightFace Model Packs
- ⚙️ Face Swap Model Installation
  # 📥 Required AI Models

FaceCore is designed to be lightweight and **does not automatically download AI models**.

You only need to download the models required for the features you plan to use.

---

# 📦 Required Models

| Model | Required | Purpose |
|--------|----------|---------|
| `buffalo_l` | ✅ Yes | Face Detection, Recognition & Embeddings |
| `inswapper_128.onnx` | Optional | Face Swapping |
| `age_googlenet.onnx` | Optional | Age Prediction |
| `gender_googlenet.onnx` | Optional | Gender Prediction |
| `emotion-ferplus-8.onnx` | Optional | Emotion Recognition |

---

# 🌐 Download Links

## 1️⃣ InsightFace Model Pack

Required for

- Face Detection
- Face Recognition
- Face Comparison
- Face Registration
- Face Embeddings

Download:

https://github.com/deepinsight/insightface/tree/master/python-package

or allow InsightFace to download it automatically on first run.

---

## 2️⃣ Face Swap Model

Required only for

- Image Face Swapping

Download

https://huggingface.co/ezioruan/inswapper_128.onnx

File

```
inswapper_128.onnx
```

---

## 3️⃣ Age Detection Model

Download

https://huggingface.co/onnxmodelzoo/age_googlenet

File

```
age_googlenet.onnx
```

---

## 4️⃣ Gender Detection Model

Download

https://huggingface.co/onnxmodelzoo/gender_googlenet

File

```
gender_googlenet.onnx
```

---

## 5️⃣ Emotion Detection Model

Download

https://huggingface.co/onnxmodelzoo/emotion-ferplus-8

File

```
emotion-ferplus-8.onnx
```

---

# 📂 Recommended Folder Structure

```text
FaceCore/
│
├── models/
│   ├── inswapper_128.onnx
│   ├── age_googlenet.onnx
│   ├── gender_googlenet.onnx
│   └── emotion-ferplus-8.onnx
│
├── examples.py
├── README.md
└── ...
```

---

# 🚀 Automatic InsightFace Download

The first time you initialize FaceCore,

```python
from FaceCore import Face

face = Face()
```

InsightFace automatically downloads the required **buffalo_l** model if it is not already installed.

No manual download is normally required.

---

# 📁 Default InsightFace Location

### Windows

```text
C:\Users\<Username>\.insightface\models\
```

---

### Linux

```text
~/.insightface/models/
```

---

### macOS

```text
~/.insightface/models/
```

---

# 📦 Manual Model Loading

You can manually specify model paths when creating the `Face` object.

### Face Swap

```python
from FaceCore import Face

face = Face(
    swap_model="models/inswapper_128.onnx"
)
```

---

### Age Detection

```python
from FaceCore import Face

face = Face(
    age_model="models/age_googlenet.onnx"
)
```

---

### Gender Detection

```python
from FaceCore import Face

face = Face(
    gender_model="models/gender_googlenet.onnx"
)
```

---

### Emotion Detection

```python
from FaceCore import Face

face = Face(
    emotion_model="models/emotion-ferplus-8.onnx"
)
```

---

### Multiple Models

```python
from FaceCore import Face

face = Face(
    swap_model="models/inswapper_128.onnx",
    age_model="models/age_googlenet.onnx",
    gender_model="models/gender_googlenet.onnx",
    emotion_model="models/emotion-ferplus-8.onnx"
)
```

---

# 📊 Feature Requirements

| Feature | Required Model |
|----------|----------------|
| Face Detection | buffalo_l |
| Face Recognition | buffalo_l |
| Face Comparison | buffalo_l |
| Face Registration | buffalo_l |
| Face Landmarks | buffalo_l |
| Face Analysis | buffalo_l |
| Age Prediction | age_googlenet.onnx |
| Gender Prediction | gender_googlenet.onnx |
| Emotion Recognition | emotion-ferplus-8.onnx |
| Face Swapping | inswapper_128.onnx |
| Webcam Recognition | buffalo_l |

---

# ⚠️ Common Model Errors

## Face Swap Model Missing

```text
FileNotFoundError: inswapper_128.onnx
```

Download the Face Swap model and provide its path using the `swap_model` parameter.

---

## Age Model Missing

```text
Age model not found
```

Download `age_googlenet.onnx` and pass it using the `age_model` parameter.

---

## Gender Model Missing

```text
Gender model not found
```

Download `gender_googlenet.onnx` and pass it using the `gender_model` parameter.

---

## Emotion Model Missing

```text
Emotion model not found
```

Download `emotion-ferplus-8.onnx` and pass it using the `emotion_model` parameter.

---

## InsightFace Downloads Every Time

If the InsightFace model is downloaded repeatedly:

- Make sure the download completed successfully.
- Verify that the `.insightface/models` directory is writable.
- Avoid deleting the model cache between runs.

---

# 💡 Best Practices

- Keep all optional ONNX models in a dedicated `models/` folder.
- Initialize the `Face()` object only once.
- Reuse the same model files across projects.
- Use absolute paths if your models are stored outside the project directory.

---

## ➡️ Part 3

Part 3 will include:

- 📚 Complete API Documentation
- 🔍 `detect()`
- 👤 `register()`
- 🆚 `compare()`
- 🪪 `recognize()`
- 📍 `landmarks()`
- 🎂 `analyze()`
- 😀 Emotion Detection Examples
- 🎭 `swap()`
- 📷 Webcam Recognition
- 📊 Return Values & Examples
  # 📚 API Documentation

Once you have initialized FaceCore,

```python
from FaceCore import Face

face = Face()
```

all features can be accessed from the same `Face` object.

---

# 🔍 Face Detection

Detect one or multiple faces inside an image.

## Syntax

```python
faces = face.detect("group.jpg")
```

---

## Example

```python
from FaceCore import Face

face = Face()

faces = face.detect("group.jpg")

print(faces)
```

---

## Returns

```python
[
    {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.99,
        "landmarks": [...],
        "embedding": [...]
    }
]
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| image | str | Image path |

---

# 🆚 Face Comparison

Compare two face images and return a similarity score.

---

## Syntax

```python
score = face.compare(
    "person1.jpg",
    "person2.jpg"
)
```

---

## Example

```python
from FaceCore import Face

face = Face()

score = face.compare(
    "me1.jpg",
    "me2.jpg"
)

print(score)
```

Output

```
97.84
```

Higher score means the faces are more similar.

---

## Parameters

| Parameter | Type |
|------------|------|
| image1 | str |
| image2 | str |

---

## Returns

```python
float
```

---

# 👤 Register Face

Register a person's face for future recognition.

---

## Syntax

```python
face.register(
    name,
    image
)
```

---

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
face.register("Emma","emma.jpg")
face.register("John","john.jpg")
```

---

## Parameters

| Parameter | Type |
|------------|------|
| name | str |
| image | str |

---

# 🪪 Face Recognition

Recognize a previously registered face.

---

## Syntax

```python
name = face.recognize(
    "person.jpg"
)
```

---

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

---

### Using a Threshold

```python
name = face.recognize(
    "person.jpg",
    threshold=70
)
```

---

## Parameters

| Parameter | Type |
|------------|------|
| image | str |
| threshold | int *(optional)* |

---

## Returns

```python
str
```

---

# 📍 Facial Landmarks

Extract facial landmark points.

---

## Syntax

```python
points = face.landmarks(
    "person.jpg"
)
```

---

## Example

```python
points = face.landmarks(
    "person.jpg"
)

print(points)
```

---

## Returns

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

## Parameters

| Parameter | Type |
|------------|------|
| image | str |

---

# 🎂 Face Analysis

Analyze a detected face.

Depending on the loaded models, FaceCore can estimate

- Age
- Gender

---

## Syntax

```python
info = face.analyze(
    "person.jpg"
)
```

---

## Example

```python
info = face.analyze(
    "person.jpg"
)

print(info)
```

Possible Output

```python
{
    "age":"25-32",
    "gender":"Male",
    "confidence":0.98
}
```

---

# 😀 Emotion Recognition

If an emotion model is loaded, FaceCore can also predict facial emotions.

---

## Syntax

```python
info = face.analyze(
    "person.jpg",
    include_emotion=True
)
```

---

## Example

```python
info = face.analyze(
    "person.jpg",
    include_emotion=True
)

print(info)
```

Possible Output

```python
{
    "age":"25-32",
    "gender":"Male",
    "emotion":"Happy",
    "confidence":0.99
}
```

---

# 🎭 Face Swapping

Swap a source face onto a target image.

---

## Syntax

```python
face.swap(
    source,
    target,
    output
)
```

---

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
---

# 🎭 Live Webcam Face Swapping

Swap your face onto detected faces in real-time using your webcam.

The source face is loaded once, and every detected face in the webcam stream is replaced with the source face.

---

## Syntax

```python
face.webcam_swap(
    source_image,
    swap_every=2
)
```

---

## Example

```python
from FaceCore import Face

face = Face(
    swap_model="inswapper_128.onnx"
)

face.webcam_swap(
    "me.jpg"
)
```

---

## Process Every Second Frame

```python
face.webcam_swap(
    "me.jpg",
    swap_every=2
)
```

This is the recommended setting and provides a good balance between performance and visual quality.

---

## Process Every Third Frame

```python
face.webcam_swap(
    "me.jpg",
    swap_every=3
)
```

Using a larger value increases FPS while slightly reducing swap frequency.

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `source_image` | str | Source face image used for swapping |
| `swap_every` | int | Swap every *N* frames (Default: `2`) |

---

## Returns

```python
None
```

The swapped webcam stream is displayed in a live OpenCV window.

Press **Q** at any time to exit.

---

## Performance Tips

| swap_every | Performance | Quality |
|------------|-------------|---------|
| `1` | Slowest | Best |
| `2` | Recommended | Excellent |
| `3` | Faster | Very Good |
| `4+` | Fastest | Lower swap frequency |

---

## Example

```python
from FaceCore import Face

face = Face(
    swap_model="inswapper_128.onnx"
)

face.webcam_swap(
    source_image="my_face.jpg",
    swap_every=2
)
```

## Parameters

| Parameter | Type |
|------------|------|
| source | str |
| target | str |
| output | str |

---

# 📷 Webcam Recognition

Run real-time face recognition from your webcam.

---

## Syntax

```python
face.webcam_recognize()
```

---

## Example

```python
from FaceCore import Face

face = Face()

face.register(
    "Srish",
    "srish.jpg"
)

face.webcam_recognize()
```

Press **Q** to exit the webcam window.

---

# 📊 API Summary

| Method | Description |
|----------|-------------|
| `detect()` | Detect faces in an image |
| `compare()` | Compare two faces |
| `register()` | Register a person's face |
| `recognize()` | Recognize registered faces |
| `landmarks()` | Extract facial landmarks |
| `analyze()` | Predict age, gender and emotion |
| `swap()` | Swap faces between images |
| `webcam_recognize()` | Live face recognition |

---

# 💡 Usage Tips

- Create only one `Face()` instance.
- Reuse the same object throughout your application.
- Register faces before calling `recognize()`.
- Load optional models only when needed.
- Store your AI models in a dedicated `models/` folder.
- Use high-quality, front-facing images for the best recognition accuracy.

---

## ➡️ Part 4

Part 4 will include:

- ⚠️ Troubleshooting
- ❓ Frequently Asked Questions (FAQ)
- 🚀 Performance Tips
- 🛣️ Roadmap
- 🤝 Contributing
- 📜 License
- ❤️ Support
- 👨‍💻 Author
- 🙏 Credits
- 📈 Changelog
- ⭐ Final GitHub Footer
  # ⚠️ Troubleshooting

If you encounter any issues while using FaceCore, check the solutions below.

---

# 📦 Installation Errors

## ModuleNotFoundError

Example

```text
ModuleNotFoundError: No module named 'insightface'
```

Install the required packages.

```bash
pip install insightface onnxruntime opencv-python pillow numpy
```

or

```bash
pip install -r requirements/requirements.txt
```

---

## ONNX Runtime Missing

```text
ModuleNotFoundError: No module named 'onnxruntime'
```

Install ONNX Runtime.

```bash
pip install onnxruntime
```

---

## OpenCV Missing

```text
ModuleNotFoundError: No module named 'cv2'
```

Install OpenCV.

```bash
pip install opencv-python
```

---

# 🤖 Model Errors

## Face Swap Model Not Found

```text
FileNotFoundError: inswapper_128.onnx
```

Download the model and specify its location.

```python
face = Face(
    swap_model="models/inswapper_128.onnx"
)
```

---

## Age Model Missing

```text
Age model not found
```

Download

```
age_googlenet.onnx
```

and load it when creating the `Face` object.

---

## Gender Model Missing

```text
Gender model not found
```

Download

```
gender_googlenet.onnx
```

---

## Emotion Model Missing

```text
Emotion model not found
```

Download

```
emotion-ferplus-8.onnx
```

---

## InsightFace Downloads Every Run

Possible reasons

- Model download was interrupted.
- Cache directory was deleted.
- No write permission to the model directory.

Delete the incomplete model folder and run FaceCore again.

---

# 📷 Webcam Problems

## Webcam Not Opening

Check the following:

- Webcam permissions are enabled.
- No other application is using the camera.
- Camera drivers are installed correctly.
- Try another camera index if multiple webcams are connected.

---

## Webcam Lag

For smoother performance

```python
face.webcam_swap(
    "me.jpg",
    swap_every=2
)
```

or

```python
face.webcam_swap(
    "me.jpg",
    swap_every=3
)
```

Avoid using

```python
swap_every=1
```

unless maximum quality is required.

---

# 👤 Recognition Problems

## Face Not Recognized

Possible causes

- Face was never registered.
- Poor lighting.
- Face is rotated.
- Face is too small.
- Recognition threshold is too high.

Try using a lower threshold.

```python
face.recognize(
    "person.jpg",
    threshold=60
)
```

---

## Wrong Person Detected

Improve accuracy by

- Registering higher-quality images.
- Using front-facing faces.
- Registering multiple images of the same person.
- Ensuring good lighting.

---

# 🎭 Face Swap Problems

## Swap Looks Incorrect

Check that

- Source image contains a clear face.
- Target image contains a visible face.
- Face is not heavily rotated.
- Face is not covered by objects.

---

## No Face Found

Possible reasons

- Blurry image.
- Very small face.
- Side profile.
- Poor lighting.
- Face outside the frame.

---

# 🚀 Performance Tips

For the best experience

- ✅ Initialize `Face()` only once.
- ✅ Reuse the same object.
- ✅ Store models on an SSD.
- ✅ Use clear, front-facing images.
- ✅ Use `swap_every=2` for webcam swapping.
- ✅ Close unused applications while using the webcam.

---

# ❓ Frequently Asked Questions

### Does FaceCore require a GPU?

No.

FaceCore works perfectly on CPU.

If ONNX Runtime GPU is installed, inference may be accelerated automatically.

---

### Can FaceCore detect multiple faces?

Yes.

Multiple faces are supported in detection, recognition, and swapping.

---

### Can I register multiple people?

Yes.

Example

```python
face.register("Alex","alex.jpg")
face.register("Emma","emma.jpg")
face.register("John","john.jpg")
```

---

### Does registration persist after restarting Python?

No.

Registered faces are stored in memory for the current session only.

---

### Which image formats are supported?

- JPG
- JPEG
- PNG
- BMP
- WEBP

---

### Does FaceCore work offline?

Yes.

Once the required AI models are downloaded, FaceCore works completely offline.

---

### Can I use my own ONNX models?

Yes.

Provide the model path while initializing `Face()`.

---

### Which operating systems are supported?

- Windows
- Linux
- macOS

---

# 🛣 Roadmap

Upcoming improvements

- GPU acceleration
- Video face swapping
- Persistent face database
- Batch face processing
- Face enhancement
- Face restoration
- Face tracking
- Face quality scoring
- Live emotion recognition improvements
- Command Line Interface (CLI)
- Docker support
- REST API
- Performance optimizations

---

# 🤝 Contributing

Contributions are always welcome.

1. Fork the repository.

2. Create a feature branch.

```bash
git checkout -b feature-name
```

3. Commit your changes.

```bash
git commit -m "Added new feature"
```

4. Push your branch.

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

# 📜 License

This project is licensed under the **MIT License**.

You are free to

- Use
- Modify
- Distribute
- Share

Please include the original license when redistributing this project.

---

# ❤️ Support

If you find FaceCore useful,

- ⭐ Star this repository
- 🐛 Report bugs
- 💡 Suggest new features
- 🤝 Contribute to the project

Your support helps improve FaceCore for everyone.

---

# 👨‍💻 Author

## **Srish Ghosh**

Python Developer • AI Enthusiast • Open Source Contributor

**GitHub**

```text
https://github.com/developer-srish
```

---

# 🙏 Credits

FaceCore is built using these amazing open-source projects.

- Python
- InsightFace
- OpenCV
- ONNX Runtime
- NumPy
- Pillow

Special thanks to the maintainers of these projects for making modern computer vision accessible to everyone.

---

# 📈 Changelog

## Version 2.2.0

### Added

- Face Detection
- Face Recognition
- Face Comparison
- Face Registration
- Facial Landmarks
- Age Prediction
- Gender Prediction
- Emotion Recognition
- Image Face Swapping
- Webcam Recognition
- Webcam Face Swapping

### Improved

- Faster inference
- Cleaner API
- Better documentation
- Simplified model loading
- Improved recognition workflow

### Fixed

- General stability improvements
- Better error handling
- Cross-platform compatibility

---

# ⭐ Support the Project

If FaceCore helps you build something awesome,

<p align="center">

## ⭐ Give this repository a Star!

**Made with ❤️ by Srish Ghosh**

</p>
  
