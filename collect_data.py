import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# CSV file setup
file_name = "gestures.csv"
file_exists = os.path.isfile(file_name)

csv_file = open(file_name, 'a', newline='')
csv_writer = csv.writer(csv_file)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press keys to collect data:")
print("h = HELLO | y = YES | n = NO | t = THANK YOU | p = HELP")
print("Press ESC to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        data_row = []

        for lm in landmarks.landmark:
            data_row.extend([lm.x, lm.y, lm.z])

        if key == ord('h'):
            data_row.append(0)
            csv_writer.writerow(data_row)
            print("HELLO captured")

        elif key == ord('y'):
            data_row.append(1)
            csv_writer.writerow(data_row)
            print("YES captured")

        elif key == ord('n'):
            data_row.append(2)
            csv_writer.writerow(data_row)
            print("NO captured")

        elif key == ord('t'):
            data_row.append(3)
            csv_writer.writerow(data_row)
            print("THANK YOU captured")

        elif key == ord('p'):
            data_row.append(4)
            csv_writer.writerow(data_row)
            print("HELP captured")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
