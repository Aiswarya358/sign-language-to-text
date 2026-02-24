# Sign Language to Text Converter

This project is a real-time sign language gesture recognition system that converts common daily hand gestures into text using computer vision and machine learning.

---

## Features
- Real-time hand gesture detection using webcam
- Hand landmark extraction using MediaPipe
- Machine learning-based gesture classification
- Converts gestures into readable text
- Supports common daily static gestures

---

## Gestures Supported
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
2. MediaPipe extracts 21 hand landmarks  
3. Landmark coordinates are converted into feature vectors  
4. KNN classifier predicts the gesture  
5. Predicted gesture is displayed as text  

---

## How to Run

### 1. Install dependencies
```bash
pip install opencv-python mediapipe numpy pandas scikit-learn
```

### 2. Train the model
```bash
python train_model.py
```

### 3. Run real-time prediction
```bash
python real_time_prediction.py
```

---

## Dataset
- Contains only numerical hand landmark coordinates
- No images or videos are stored
- Data collected using webcam

---

## Limitations
- Works with predefined static gestures only
- Sensitive to lighting conditions
- Single-hand detection

---

## Author
Aiswarya K Rejikumar
