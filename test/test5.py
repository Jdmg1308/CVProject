#@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2
import mediapipe as mp
import keyboard

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    for gesture in result.gestures:
            top_gesture = [category.category_name for category in gesture][0]
            print(top_gesture)


# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    min_tracking_confidence=0.7,
    running_mode = VisionRunningMode.LIVE_STREAM, result_callback = print_result)
recognizer = vision.GestureRecognizer.create_from_options(options)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands
top_gesture = "none"

# Set the desired window size
window_width = 1080  # Change this to your desired width
window_height = 720  # Change this to your desired height

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

hands = mphands.Hands()
while True:
    data, frame=cap.read()
    if not data:
        break
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 8, (0, 0, 0), -1)  # black color (BGR format)
                cv2.circle(frame, (x, y), 6, (12, 124, 194), -1)  # white color (BGR format)

            for connection in mphands.HAND_CONNECTIONS:
                x0, y0 = int(hand_landmarks.landmark[connection[0]].x * frame.shape[1]), int(hand_landmarks.landmark[connection[0]].y * frame.shape[0])
                x1, y1 = int(hand_landmarks.landmark[connection[1]].x * frame.shape[1]), int(hand_landmarks.landmark[connection[1]].y * frame.shape[0])
                cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 0), 4)  # black color (BGR format)

                cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)  # white color (BGR format)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = recognizer.recognize_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    # STEP 5: Process the result. In this case, visualize it.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # top_gesture = recognition_result.gestures
    recognition_result
    cv2.imshow("Handtracker", frame)
    cv2.waitKey(1)

    if(keyboard.is_pressed('q')):
        break

cap.release()

cv2.destroyAllWindows()