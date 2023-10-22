#@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import keyboard
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import threading

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
index_x = 0
index_y = 0
mphands = mp.solutions.hands
top_gesture = 'None'
screen_width, screen_height = pyautogui.size()
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def jesters():
    # print(top_gesture)
    match top_gesture:
        case 'Pointing_Up':
            mouse_click(index_x, index_y)
        # case 'Closed_Fist':
            # print("Closed_Fist")

def mouse_click(position_x, position_y):
    pyautogui.click(position_x, position_y)
    pyautogui.sleep(1)

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global top_gesture
    for gesture in result.gestures:
            top_gesture = [category.category_name for category in gesture][0]
            # print(top_gesture)

# Moves the mouse to the index finger
def move_mouse(results, frame, width, height):
    global index_x
    global index_y
    global screen_width
    global screen_height
    index = results.multi_hand_landmarks[0].landmark[8]
    x = int(index.x * width)
    y = int(index.y * height)
    cv2.circle(img=frame, center=(x,y), radius = 10, color=(100,100,100))
    index_x = screen_width/width * x
    index_y = screen_height/height * y
    pyautogui.moveTo(index_x, index_y)
    time.sleep(.005)

#Nathan was straight cooking on this method.
def cooking(results, frame):
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 8, (0, 0, 0), -1)  # black color (BGR format)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)  # white color (BGR format)
        for connection in mphands.HAND_CONNECTIONS:
            x0, y0 = int(hand_landmarks.landmark[connection[0]].x * frame.shape[1]), int(hand_landmarks.landmark[connection[0]].y * frame.shape[0])
            x1, y1 = int(hand_landmarks.landmark[connection[1]].x * frame.shape[1]), int(hand_landmarks.landmark[connection[1]].y * frame.shape[0])
            cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 0), 4)  # black color (BGR format)            
            cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)  # white color (BGR format)

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options, min_tracking_confidence=0.7,
    running_mode = VisionRunningMode.LIVE_STREAM, result_callback = print_result)
recognizer = vision.GestureRecognizer.create_from_options(options)

def THE_method(width, height):
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    hands = mphands.Hands()

    while True:
        data, frame = cap.read()
        if not data:
            break
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        
        if results.multi_hand_landmarks:
            move_mouse(results, frame, width, height)
            cooking(results, frame)
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognition_result = recognizer.recognize_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        jesters()
        cv2.imshow("Handtracker", frame)
        cv2.waitKey(1)
        if(keyboard.is_pressed('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

def gen_hand():
    global screen_width
    global screen_height
    ratio = screen_height/screen_width
    window_width = 1080
    window_height = ratio * window_width
    THE_method(window_width, window_height)

if __name__ == "__main__":
    gen_hand()
    