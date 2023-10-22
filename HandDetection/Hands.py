import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

# Set the desired window size
window_width = 1080  # Change this to your desired width
window_height = 720  # Change this to your desired height

# OpenCV window settings
#cv2.resizeWindow("Handtracker", window_width, window_height)

cap = cv2.VideoCapture(0)
hands = mphands.Hands()
while True:
    data, image=cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 8, (0, 0, 0), -1)  # black color (BGR format)
                cv2.circle(image, (x, y), 6, (255, 255, 255), -1)  # white color (BGR format)

            for connection in mphands.HAND_CONNECTIONS:
                x0, y0 = int(hand_landmarks.landmark[connection[0]].x * image.shape[1]), int(hand_landmarks.landmark[connection[0]].y * image.shape[0])
                x1, y1 = int(hand_landmarks.landmark[connection[1]].x * image.shape[1]), int(hand_landmarks.landmark[connection[1]].y * image.shape[0])
                cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), 6)  # black color (BGR format)

                cv2.line(image, (x0, y0), (x1, y1), (255, 255, 255), 4)  # white color (BGR format)

    #cv2.namedWindow("Handtracker", cv2.WINDOW_NORMAL)
    cv2.imshow("Handtracker", image)
    #cv2.resizeWindow("Handtracker", window_width, window_height)
    cv2.waitKey(1)