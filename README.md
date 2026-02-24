# Sign Language to Text Converter

A real-time Sign Language to Text conversion system built using Computer Vision and Machine Learning.  
This project detects static hand gestures using MediaPipe hand landmarks and classifies them into predefined text outputs.

---

## Project Overview

This system captures live video from a webcam, extracts 21 hand landmarks using MediaPipe, converts them into numerical feature vectors, and classifies gestures using a K-Nearest Neighbors (KNN) model.

The predicted gesture is displayed instantly as readable text.

---

## Model Details

- Algorithm: K-Nearest Neighbors (KNN)
- Dataset Size: 602 samples
- Features per sample: 63 (21 landmarks × x, y, z)
- Number of Classes: 5
- Accuracy: 91.7% (train-test split evaluation)

---

## Supported Gestures

- Hello  
- Yes  
- No  
- Thank You  
- Help  

---

## Technologies Used

- Python  
- OpenCV  
- MediaPipe  
- Scikit-learn  
- NumPy  
- Pandas  

---

## System Workflow

1. Webcam captures live video frames  
2. MediaPipe detects the hand and extracts 21 landmarks  
3. Landmark coordinates are flattened into a feature vector  
4. KNN classifier predicts the gesture  
5. Predicted gesture is displayed in real-time  

---

## Installation

### Install Required Libraries

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn
```

---

## Usage

### Train the Model

```bash
python train_model.py
```

### Run Real-Time Prediction

```bash
python real_time_prediction.py
```

---

## Dataset Information

- Custom dataset collected using webcam  
- 602 labeled samples  
- Each sample contains 63 landmark features  
- Stored in `gestures.csv`  

---

## Limitations

- Supports only static gestures  
- Works best with single-hand detection  
- Performance depends on lighting conditions  

---

## Future Improvements

- Add dynamic gesture recognition  
- Experiment with deep learning models  
- Improve classification accuracy  
- Deploy as a web or mobile application  

---

## Author

Aiswarya K Rejikumar  
B.Tech Computer Science Engineering
