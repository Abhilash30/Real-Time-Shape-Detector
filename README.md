# Real-Time Shape Detector

/home/abhilash/Projects/shape_detector/src/captures/frame_20260923_154535_800696.png



## Overview

A real-time geometric shape detector built using OpenCV and
scikit-learn.

The project processes webcam frames, extracts contours and
handcrafted geometric features, and uses a K-Nearest Neighbors
classifier to recognize shapes.

## Features

- Real-time webcam detection
- Grayscale conversion
- Otsu thresholding
- Morphological processing
- Contour detection
- Contour filtering
- Geometric feature extraction
- Feature standardization
- KNN classification
- Bounding-box visualization
- Dataset collection and analysis
- Model evaluation

## Pipeline

Webcam
→ Grayscale
→ Threshold
→ Morphology
→ Contours
→ Feature Extraction
→ StandardScaler
→ KNN
→ Prediction

## Features Used

- Area
- Perimeter
- Number of vertices
- Aspect ratio
- Circularity

## Machine Learning

The project uses K-Nearest Neighbors with:

- `n_neighbors = 3`
- `StandardScaler`
- 80/20 train-test split
- fixed `random_state = 42`

The classifier achieved approximately 97% accuracy
on the held-out test set.



## Installation
Clone repo 
```bash
run pip install -r requirements.txt

## Running
python3 src/main.py


### Train the model
```bash
python3 src/KNN_sklearn.py
```
Limitations
Performance depends on contour quality.
Effects of morphology on shapes.
Lighting and background affect thresholding.
Hand-drawn shapes can produce noisy contours.
KNN performance depends on the quality and distribution
of the training dataset.

Future Improvements
Compare KNN with SVM, Random Forest and Decision Tree
Improve contour preprocessing
Add confidence/nearest-neighbor distance visualization
Expand the dataset
Detect more complex shapes

