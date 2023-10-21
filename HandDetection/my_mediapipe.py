import cv2
import time
import mediapipe as mp

mp_hands = mp.solutions.holistic
hands = mp_hands.Holistic(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5)

width = 1080
height = 720

# Open the video capture from your webcam (change the argument to the video file path if you want to process a video file).
cap = cv2.VideoCapture(0)

cap.set(3, width)
cap.set(4, height)

p_time = 0

while True:
    success, img = cap.read()

    c_time = time.time()

    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)

    # Convert the captured frame to RGB format.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks from the input frame.
    results = hands.process(img_rgb)

    # Process the classification result and visualize it.
    
    annotated_image = img.copy()
    mp.solutions.drawing_utils.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Landmarks", annotated_image)

    if cv2.waitKey(1) & 0xFF == 'q':  # Press Esc to exit the loop.
        break

# Release the video capture and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
