import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

# Load trained model
model = joblib.load("model.pkl")

gesture_map = {
    0: "HELLO",
    1: "YES",
    2: "NO",
    3: "THANK YOU",
    4: "HELP"
}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

prediction_queue = deque(maxlen=7)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture_text = ""

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])

        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)[0]

        prediction_queue.append(prediction)

        if len(prediction_queue) == prediction_queue.maxlen:
            most_common = Counter(prediction_queue).most_common(1)[0][0]
            gesture_text = gesture_map[most_common]

    cv2.putText(
        frame,
        f"Gesture: {gesture_text}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language to Text", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
