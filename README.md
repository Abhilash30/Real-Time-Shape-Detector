#Real-Time Shape Detector

### A classical computer vision pipeline for detecting and classifying geometric shapes from a webcam using OpenCV

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange?logo=numpy)
![Status](https://img.shields.io/badge/Status-Pre--ML%20Prototype-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **A from-scratch classical computer vision project that transforms webcam images into geometric features and uses those features to identify shapes in real time.**

---

## 📌 Overview

**Real-Time Shape Detector** is a computer vision project built using Python, OpenCV, and NumPy.

The system captures frames from a webcam and processes them through a sequence of classical computer vision operations:

```text
Webcam
   ↓
BGR Image
   ↓
Grayscale Conversion
   ↓
Binary Thresholding
   ↓
Morphological Cleaning
   ↓
Contour Detection
   ↓
Geometric Feature Extraction
   ↓
Rule-Based Classification
   ↓
Detected Shape
```

The current implementation focuses entirely on **classical computer vision**.

There is deliberately **no machine learning model yet**.

The goal of this stage of the project is to understand how much can be achieved using image processing, contours, geometry, and manually designed classification rules before introducing ML.

---

# 🎯 Project Goals

This project was built as a practical way to learn the fundamentals of computer vision by implementing a complete pipeline rather than immediately relying on a pre-trained model.

The main goals are:

* Understand how images are represented computationally.
* Work with OpenCV's image-processing operations.
* Convert real-world camera frames into useful binary representations.
* Understand noise removal using morphology.
* Understand contour detection.
* Extract meaningful geometric features from objects.
* Build a rule-based classifier.
* Understand the limitations of manually designed computer vision systems.
* Establish a foundation for eventually replacing the rule-based classifier with machine learning.

---

# ✨ Features

### Current capabilities

* 📷 Real-time webcam input
* ⚫ Grayscale image conversion
* ⚪ Binary thresholding
* 🧹 Morphological noise removal
* 🔍 External contour detection
* 📐 Contour area calculation
* 📏 Contour perimeter calculation
* 🔺 Polygon approximation
* 📦 Bounding-box extraction
* ↔️ Aspect-ratio calculation
* ⭕ Circularity calculation
* 🧠 Rule-based shape classification
* 🖥️ Real-time visualization of detected contours and labels

### Currently recognized shapes

* Triangle
* Square
* Circle
* Unknown / unsupported shapes

---

# 🧠 How It Works

The detector is based on a simple principle:

> **Convert visual information into geometric information.**

Instead of trying to understand the image directly, the system progressively simplifies it.

For example:

```text
                    REAL WORLD

                       ✏️
              Hand-drawn shapes
                       │
                       ▼

                 📷 CAMERA FRAME
                       │
                       ▼

              ┌─────────────────┐
              │   GRAYSCALE     │
              │                 │
              │ Removes color   │
              │ information     │
              └─────────────────┘
                       │
                       ▼

              ┌─────────────────┐
              │   THRESHOLD     │
              │                 │
              │ Black / White   │
              └─────────────────┘
                       │
                       ▼

              ┌─────────────────┐
              │   MORPHOLOGY    │
              │                 │
              │ Remove noise    │
              │ Fill gaps       │
              └─────────────────┘
                       │
                       ▼

              ┌─────────────────┐
              │    CONTOURS     │
              │                 │
              │ Object borders  │
              └─────────────────┘
                       │
                       ▼

              ┌─────────────────┐
              │    FEATURES     │
              │                 │
              │ Area            │
              │ Perimeter       │
              │ Vertices        │
              │ Aspect Ratio    │
              │ Circularity     │
              └─────────────────┘
                       │
                       ▼

              ┌─────────────────┐
              │   CLASSIFIER    │
              │                 │
              │ Hand-written    │
              │ decision rules  │
              └─────────────────┘
                       │
                       ▼

                 🔺 TRIANGLE
                 ⬜ SQUARE
                 ⭕ CIRCLE
```

Each stage reduces the complexity of the problem.

---

# 🏗️ Software Architecture

The project follows a modular architecture where each file is responsible for a specific part of the computer vision pipeline.

```text
shape_detector/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── webcam.py
│   ├── preprocessing.py
│   ├── contours.py
│   ├── features.py
│   ├── classifier.py
│   └── main.py
│
├── models/
│
└── README.md
```

## Module Responsibilities

### `webcam.py`

Responsible only for camera operations.

```text
Camera
  │
  ├── Open
  ├── Read frame
  ├── Save frame
  └── Release
```

The module does not perform image processing.

---

### `preprocessing.py`

Contains image-processing operations.

Current responsibilities include:

* Grayscale conversion
* Thresholding
* Morphological operations

Conceptually:

```text
BGR
 ↓
Grayscale
 ↓
Binary
 ↓
Clean Binary
```

Keeping preprocessing separate makes the pipeline easier to test and modify.

---

### `contours.py`

Responsible for detecting object boundaries.

```text
Clean Binary
      ↓
findContours()
      ↓
List of contours
```

The current implementation focuses on external contours because the detector is interested primarily in identifying individual visible shapes.

---

### `features.py`

Responsible for converting contours into numerical measurements.

Current features include:

```text
Contour
   │
   ├── Area
   ├── Perimeter
   ├── Polygon approximation
   ├── Vertex count
   ├── Aspect ratio
   └── Circularity
```

This module is particularly important because it creates the bridge between **computer vision and machine learning**.

---

### `classifier.py`

Contains the current manually designed classification logic.

The classifier receives geometric measurements and returns a predicted shape:

```text
Features
   ↓
Rule-Based Decision
   ↓
Shape Label
```

The classifier currently uses heuristics such as:

* Number of polygon vertices
* Circularity
* Aspect ratio

---

### `main.py`

Acts as the pipeline coordinator.

It does not contain the underlying image-processing algorithms.

Its responsibility is to connect the modules:

```text
Webcam
   ↓
Preprocessing
   ↓
Contours
   ↓
Features
   ↓
Classifier
   ↓
Visualization
```

This keeps the project modular and makes individual stages easier to replace later.

---

# 🔄 Data Flow

The complete data flow is:

```text
┌──────────────┐
│    Webcam    │
└──────┬───────┘
       │
       ▼
┌────────────────┐
│   BGR Frame    │
│  H × W × 3     │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   Grayscale    │
│     H × W      │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   Threshold    │
│   Binary Image │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   Morphology   │
│ Clean Binary   │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│    Contours    │
│ [C₁, C₂, ...]  │
└───────┬────────┘
        │
        ▼
┌─────────────────────────┐
│   Feature Extraction    │
│                         │
│ Area                    │
│ Perimeter               │
│ Vertices                │
│ Aspect Ratio            │
│ Circularity             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Rule-Based Classifier │
└───────────┬─────────────┘
            │
            ▼
      Shape Prediction
```

---

# 🔬 Stage 1 — Webcam Capture

The first stage obtains a continuous stream of frames from the camera.

Each frame is initially represented as a color image in OpenCV's BGR format.

```text
Camera
  ↓
Frame
  ↓
BGR image
```

The webcam module handles:

* Camera initialization
* Frame acquisition
* Frame saving
* Camera release

This isolates hardware interaction from the rest of the application.

---

# ⚫ Stage 2 — Grayscale Conversion

Color is unnecessary for the current problem.

The detector is primarily interested in:

> **Where is the shape and where is the background?**

Therefore, the BGR image is converted into grayscale.

```text
BGR Image
   ↓
Grayscale Image
```

This reduces the information from three color channels to a single intensity channel.

The conversion is performed using OpenCV:

```python
cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

The grayscale image is then passed to thresholding.

---

# ⚪ Stage 3 — Binary Thresholding

The grayscale image contains continuous intensity values.

For shape detection, we want a much simpler representation:

```text
Background → Black
Shape      → White
```

This is achieved using thresholding.

Conceptually:

```text
Pixel < threshold
       ↓
     BLACK

Pixel ≥ threshold
       ↓
     WHITE
```

The result is a binary image.

```text
Grayscale
    ↓
Threshold
    ↓
Binary
```

This makes the shapes easier to isolate from the background.

---

# 🧹 Stage 4 — Morphological Cleaning

Real camera images aren't perfect.

The thresholded image can contain:

* Small noise
* Holes
* Broken boundaries
* Small gaps
* Unwanted isolated pixels

Morphological operations are used to clean the binary image.

The main operations used are:

### Erosion

Shrinks white regions.

```text
█████
█████
█████

     ↓

 ███
 ███
```

Useful for removing small unwanted regions.

---

### Dilation

Expands white regions.

```text
 ███
 ███

  ↓

█████
█████
█████
```

Useful for reconnecting nearby regions and filling small gaps.

---

### Opening

```text
Erosion
   ↓
Dilation
```

Primarily useful for removing small isolated noise.

---

### Closing

```text
Dilation
   ↓
Erosion
```

Useful for closing small gaps and holes.

The goal is not to produce a mathematically perfect binary image.

The goal is to produce:

> **A clean, connected representation of the shapes suitable for contour detection.**

---

# 🔍 Stage 5 — Contour Detection

Once the binary image has been cleaned, contours are extracted.

A contour represents a boundary around a connected region.

```text
Binary Image

     █████
   ██     ██
  █         █
  █         █
   ██     ██
     █████

        ↓

     Contour
```

OpenCV's `findContours()` is used to extract these boundaries.

For this project, external contours are used because the primary objective is identifying the outer boundary of each shape.

The result is conceptually:

```text
Clean Binary
      ↓
findContours()
      ↓
[
   contour_1,
   contour_2,
   contour_3,
   ...
]
```

Each contour can now be analyzed independently.

---

# 📐 Stage 6 — Feature Extraction

This is where the contour becomes numerical data.

Instead of thinking:

> "This looks like a square."

the program can measure:

```text
Area          = 15,234
Perimeter     = 462.8
Vertices      = 4
Aspect Ratio  = 1.03
Circularity   = 0.78
```

These measurements form a **feature representation** of the shape.

---

## Area

The area measures how much space is enclosed by the contour.

```python
cv2.contourArea(contour)
```

Area is useful for filtering out tiny noise.

However, it is not a particularly good standalone classification feature because it changes when the same shape is drawn at a different scale.

```text
Small square       Large square

   ████            █████████
   ████            █████████
   ████            █████████
```

Same shape, different area.

---

## Perimeter

The perimeter measures the length of the contour.

```python
cv2.arcLength(contour, True)
```

Like area, perimeter changes with the scale of the object.

It is nevertheless useful when combined with other measurements.

---

## Polygon Approximation

Real contours contain many points.

For example:

```text
Contour:

• • • • • • • • • • •
 •                 •
  •               •
   • • • • • • • •
```

A triangle may contain dozens or hundreds of contour points.

`approxPolyDP()` simplifies that contour into a smaller polygon.

```text
Many contour points
        ↓
   approxPolyDP()
        ↓
      /\
     /  \
    /____\
```

The resulting number of vertices becomes an important feature.

For example:

```text
Triangle → approximately 3 vertices
Square   → approximately 4 vertices
Circle   → many vertices
```

The amount of simplification is controlled using the `epsilon` parameter.

---

# ↔️ Aspect Ratio

The contour's bounding rectangle provides:

```text
x, y, width, height
```

The aspect ratio is:

$$
Aspect\ Ratio = \frac{width}{height}
$$

For a square:

```text
width ≈ height

aspect ratio ≈ 1
```

For a wide rectangle:

```text
width > height

aspect ratio > 1
```

This helps distinguish shapes with similar vertex counts.

---

# ⭕ Circularity

Circularity measures how closely a shape resembles a circle.

The formula is:

$$
C = \frac{4\pi A}{P^2}
$$

where:

* \(A\) = area
* \(P\) = perimeter

A perfect circle has:

$$
C = 1
$$

Other shapes generally produce smaller values.

Conceptually:

```text
More circular
     ↓
Higher circularity

Less circular
     ↓
Lower circularity
```

This provides a useful way to distinguish circles from polygons.

---

# 🧠 Stage 7 — Rule-Based Classification

The current classifier does not learn.

Instead, it uses manually designed rules.

Conceptually:

```text
              Features
                  │
       ┌──────────┼──────────┐
       │          │          │
    Vertices  Circularity  Aspect Ratio
       │          │          │
       └──────────┼──────────┘
                  ↓
           Decision Rules
                  ↓
             Prediction
```

A simplified version of the decision process is:

```text
             Number of vertices
                     │
          ┌──────────┼──────────┐
          │          │          │
         3           4       Many vertices
          │          │          │
          ▼          ▼          ▼
      Triangle     Square    Circularity
                                  │
                              High → Circle
```

The current system therefore represents a traditional **hand-engineered computer vision approach**.

---

# 🧪 Why Accuracy Is Limited

The current classifier is intentionally simple, and its imperfect accuracy is expected.

Several factors affect performance.

### 1. Hand-drawn shapes are inconsistent

Two people can draw the same shape differently.

```text
Ideal circle        Hand-drawn circle

    ○                  ◯
                      / \
```

The second shape may not have the same geometric properties as the first.

---

### 2. Contour approximation depends on epsilon

Changing `epsilon` changes the number of vertices produced by `approxPolyDP()`.

For example:

```text
Small epsilon
     ↓
Many vertices

Large epsilon
     ↓
Fewer vertices
```

Finding a good value is therefore important.

---

### 3. Lighting affects thresholding

Different lighting conditions can produce different binary images.

```text
Good lighting
     ↓
Clean binary
     ↓
Good contours
```

versus:

```text
Poor lighting
     ↓
Noisy binary
     ↓
Poor contours
     ↓
Incorrect classification
```

---

### 4. Shapes are not perfectly geometric

A hand-drawn square might actually look like:

```text
  ______
 /      |
|       |
|______/
```

Its aspect ratio and angles may differ from a perfect square.

---

### 5. Feature rules are manually selected

The classifier currently relies on thresholds chosen by the developer.

For example:

```text
circularity > X → Circle
vertices == 3   → Triangle
vertices == 4   → Square
```

These thresholds aren't learned from data.

That is the fundamental limitation of the current approach.

---

# 📊 Current Classification Strategy

| Feature      | Triangle |   Square |   Circle |
| ------------ | -------: | -------: | -------: |
| Vertices     |       ~3 |       ~4 |     Many |
| Circularity  |      Low |   Medium |     High |
| Aspect Ratio | Variable |       ~1 |       ~1 |
| Area         | Variable | Variable | Variable |
| Perimeter    | Variable | Variable | Variable |

The important observation is that some features are **scale-dependent**, while others are much more stable.

For example:

```text
                    Scale changes
                         ↓

Area             ❌ Changes significantly
Perimeter        ❌ Changes
Vertices         ✅ Mostly stable
Aspect Ratio     ✅ Mostly stable
Circularity      ✅ Mostly stable
```

This distinction will become important when machine learning is introduced.

---

# 📁 Project Structure

```text
shape_detector/
│
├── data/
│   ├── raw/
│   │   └── # Captured images
│   │
│   └── processed/
│       └── # Processed data
│
├── src/
│   ├── webcam.py
│   │   └── Camera input/output
│   │
│   ├── preprocessing.py
│   │   └── Grayscale, thresholding, morphology
│   │
│   ├── contours.py
│   │   └── Contour detection
│   │
│   ├── features.py
│   │   └── Geometric feature extraction
│   │
│   ├── classifier.py
│   │   └── Rule-based classification
│   │
│   └── main.py
│       └── Application pipeline
│
├── models/
│   └── # Reserved for future ML models
│
└── README.md
```

---

# ⚙️ Installation

## Prerequisites

* Python 3.x
* A working webcam
* OpenCV
* NumPy

Install the dependencies:

```bash
pip install opencv-python numpy
```

---

# 🚀 Running the Project

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/shape-detector.git
```

Enter the project directory:

```bash
cd shape-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the detector:

```bash
python src/main.py
```

Make sure your webcam is accessible to the operating system.

---

# 🎮 Usage

Once the program starts:

1. Point the webcam at a surface containing a shape.
2. Draw or place a recognizable geometric shape in view.
3. The frame is processed in real time.
4. The system extracts contours.
5. Features are calculated for each sufficiently large contour.
6. The rule-based classifier predicts the shape.
7. The detected contour and label are displayed.

Typical output:

```text
┌─────────────────────────────────┐
│                                 │
│          Triangle               │
│             /\                  │
│            /  \                 │
│           /____\                │
│                                 │
│                Square           │
│              ┌────┐             │
│              │    │             │
│              └────┘             │
│                                 │
└─────────────────────────────────┘
```

---

# 🧩 Design Principles

The project intentionally follows several software engineering principles.

### Separation of concerns

Each module has one primary responsibility.

```text
webcam.py
    ↓
camera handling

preprocessing.py
    ↓
image processing

contours.py
    ↓
boundary detection

features.py
    ↓
measurements

classifier.py
    ↓
decision logic

main.py
    ↓
orchestration
```

This makes the project easier to debug and extend.

---

### Pipeline-based architecture

Each stage transforms the output of the previous stage.

```text
Image
 ↓
Image
 ↓
Image
 ↓
Contours
 ↓
Features
 ↓
Prediction
```

This also makes debugging easier because every intermediate representation can be visualized.

---

### Minimal abstraction

The project avoids unnecessary classes and complex frameworks.

The purpose is to make the underlying computer vision operations explicit and understandable.

---

# 🛠️ Troubleshooting

### No camera detected

Check that your webcam is connected and available.

If multiple cameras exist, the camera index may need to be changed.

---

### Too many contours

Possible causes:

* Excessive image noise
* Poor lighting
* Threshold too low/high
* Insufficient morphological cleaning

Try adjusting the threshold or morphology parameters.

---

### Shape isn't detected

Possible causes:

* Shape is too small
* Contour is broken
* Binary image is poor
* Minimum contour area is too high

Inspect the binary and cleaned images before changing the classifier.

---

### Circle classified incorrectly

Possible causes:

* Circularity threshold is inappropriate
* Contour approximation is too aggressive
* Hand-drawn circle is irregular
* Lighting/noise distorted the contour

Inspect:

```text
Vertices
Circularity
Perimeter
Area
```

before changing the rule.

---

# 📈 Experiments

The project was developed incrementally rather than as a single large implementation.

### Experiment 1 — Grayscale

Tested the conversion from color frames to single-channel grayscale images.

### Experiment 2 — Thresholding

Converted grayscale images into binary representations.

### Experiment 3 — Morphology

Experimented with:

* Erosion
* Dilation
* Opening
* Closing
* Different kernel sizes

### Experiment 4 — Contours

Tested external contour detection on multiple shapes.

### Experiment 5 — Feature Extraction

Measured:

* Area
* Perimeter
* Vertices
* Aspect ratio
* Circularity

### Experiment 6 — Rule-Based Classification

Combined the extracted features into manually designed classification rules.

The final result works reasonably on clean, clearly drawn shapes but degrades under less controlled conditions.

---

# 🧪 What This Project Demonstrates

This project demonstrates practical understanding of several fundamental computer vision concepts:

* Image acquisition
* Image representation
* Color-space conversion
* Thresholding
* Binary image processing
* Morphological operations
* Contour detection
* Polygon approximation
* Geometric measurements
* Feature extraction
* Rule-based classification
* Real-time computer vision pipelines
* Modular software architecture

More importantly, the project demonstrates the complete transformation:

```text
Raw Pixels
    ↓
Processed Pixels
    ↓
Objects
    ↓
Measurements
    ↓
Decisions
```

---

# 🚧 Limitations

The current system is intentionally constrained.

It performs best when:

* Shapes are clearly separated.
* Lighting is reasonably consistent.
* Shapes have high contrast against the background.
* Shapes are relatively large.
* Shapes are approximately geometric.
* The camera is reasonably stable.

It is less reliable when:

* Shapes overlap.
* Lighting changes significantly.
* Backgrounds are complex.
* Shapes are heavily distorted.
* Multiple contours belong to the same object.
* Shapes are extremely small.
* Hand-drawn shapes are highly irregular.

These limitations are not just implementation problems—they motivate the next stage of the project.

---

# 🗺️ Roadmap

## ✅ Phase 1 — Webcam Capture

* [x] Open webcam
* [x] Read frames
* [x] Display frames
* [x] Save frames

## ✅ Phase 2 — Grayscale

* [x] Convert BGR → grayscale
* [x] Visualize grayscale output

## ✅ Phase 3 — Thresholding

* [x] Global thresholding
* [x] Generate binary image
* [x] Visualize threshold output

## ✅ Phase 4 — Morphology

* [x] Erosion
* [x] Dilation
* [x] Opening
* [x] Closing
* [x] Clean binary image

## ✅ Phase 5 — Contours

* [x] Detect contours
* [x] Use external contours
* [x] Visualize detected contours

## ✅ Phase 6 — Feature Extraction

* [x] Area
* [x] Perimeter
* [x] Polygon approximation
* [x] Vertex count
* [x] Aspect ratio
* [x] Circularity

## ✅ Phase 7 — Rule-Based Classification

* [x] Design classification rules
* [x] Classify triangles
* [x] Classify squares
* [x] Classify circles
* [x] Display predictions
* [x] Test failure cases

## 🔜 Phase 8 — Dataset Creation

* [ ] Capture many examples
* [ ] Extract feature vectors
* [ ] Assign labels
* [ ] Store feature data
* [ ] Analyze the dataset
* [ ] Visualize feature distributions

## 🔜 Phase 9 — Machine Learning

Replace manually designed rules with a learned classifier.

Potential models:

* [ ] Logistic Regression
* [ ] K-Nearest Neighbors
* [ ] Decision Tree
* [ ] Random Forest
* [ ] SVM

## 🔜 Phase 10 — Evaluation

* [ ] Train/validation/test split
* [ ] Accuracy
* [ ] Precision
* [ ] Recall
* [ ] F1-score
* [ ] Confusion matrix
* [ ] Analyze failure cases

## 🔜 Phase 11 — Real-Time ML

```text
Webcam
   ↓
Computer Vision Pipeline
   ↓
Contour
   ↓
Features
   ↓
Trained ML Model
   ↓
Prediction
```

The eventual goal is to compare:

```text
Hand-written rules
        VS
Machine learning
```

under the same conditions.

---

# 🔮 Future Architecture

The current architecture deliberately leaves room for an ML classifier.

### Current

```text
Features
   ↓
Hand-written rules
   ↓
Prediction
```

### Future

```text
Features
   ↓
Trained ML Model
   ↓
Prediction
```

This means most of the computer vision pipeline can remain unchanged.

Only the decision-making component needs to be replaced.

That is one of the main architectural goals of the project.

---

# 📚 Key Concepts Learned

| Concept                   | Purpose                                          |
| ------------------------- | ------------------------------------------------ |
| Grayscale                 | Reduce image complexity                          |
| Thresholding              | Separate foreground from background              |
| Morphology                | Clean binary images                              |
| Contours                  | Represent object boundaries                      |
| `approxPolyDP()`          | Simplify contour geometry                        |
| Area                      | Measure enclosed region                          |
| Perimeter                 | Measure boundary length                          |
| Aspect Ratio              | Describe width/height relationship               |
| Circularity               | Measure how circular an object is                |
| Feature Vector            | Represent an object numerically                  |
| Rule-Based Classification | Make decisions using manually defined conditions |

---

# 💡 Why Build This Before ML?

A major objective of this project is to avoid treating machine learning as a black box.

Before ML:

```text
Image
 ↓
Understand image
 ↓
Extract useful information
 ↓
Manually design decision rules
```

After ML:

```text
Image
 ↓
Understand image
 ↓
Extract useful information
 ↓
Give examples to model
 ↓
Model learns decision boundaries
```

The classical version therefore provides a useful baseline.

Instead of simply saying:

> "I trained a classifier and it predicts circles."

the project can eventually demonstrate:

> "I built the entire vision pipeline, extracted geometric features, implemented a rule-based baseline, measured its limitations, created a dataset, and then trained an ML classifier to learn the decision boundaries."

That progression is the core learning objective of this project.

---

# 🤝 Contributing

Contributions, improvements, and experiments are welcome.

A useful contribution should generally:

1. Solve a specific problem.
2. Keep modules focused on their responsibilities.
3. Avoid unnecessary dependencies.
4. Include a clear explanation of the change.
5. Preserve the readability of the computer vision pipeline.

For larger changes, open an issue first to discuss the proposed approach.

---


    

