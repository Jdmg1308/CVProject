import cv2
import time
import mediapipe as mp


width = 1080
height = 720
cap = cv2.VideoCapture(0)

cap.set(3, width)
cap.set(4, height)

p_time = 0


while True:
    success, img = cap.read()

    c_time = time.time()

    
    fps = 1 / (c_time - p_time)

    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)},', (40, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
    cv2.imshow("img", img)

    cv2.waitKey(1)

