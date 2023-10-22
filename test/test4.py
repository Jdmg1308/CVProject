import cv2
import mediapipe as mp
import numpy as np
import keyboard
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mphands = mp.solutions.hands

# Set the desired window size
window_width = 1080  # Change this to your desired width
window_height = 720  # Change this to your desired height

cap = cv2.VideoCapture(0)
cap.set(3, window_width)
cap.set(4, window_height)

hands = mphands.Hands()

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

while True:
    data, image=cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                #coloring connections
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 8, (0, 0, 0), -1)  # black color (BGR format)
                cv2.circle(image, (x, y), 6, (12, 124, 194), -1)  # white color (BGR format)
            for connection in mphands.HAND_CONNECTIONS:
                #coloring connections
                x0, y0 = int(hand_landmarks.landmark[connection[0]].x * image.shape[1]), int(hand_landmarks.landmark[connection[0]].y * image.shape[0])
                x1, y1 = int(hand_landmarks.landmark[connection[1]].x * image.shape[1]), int(hand_landmarks.landmark[connection[1]].y * image.shape[0])
                cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), 4)  # black color (BGR format)
                cv2.line(image, (x0, y0), (x1, y1), (255, 255, 255), 2)  # white color (BGR format)

    # STEP 4: Recognize gestures in the input image.
    # recognition_result = recognizer.recognize(image)

    cv2.imshow("Handtracker", image)
    cv2.waitKey(1)

    if(keyboard.is_pressed('q')):
        break

cap.release()

cv2.destroyAllWindows()