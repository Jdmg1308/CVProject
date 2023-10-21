

## RAAAAWWWRRRRRRRRR *EAGLE* *EAGLE* *EAGLE* *EAGLE* *EAGLE* *EAGLE*  merica


# what does import cv2 do?
    # does the video capture
        # while loop to continue the capture
    # .read, .imshow, .waitkey
        #.read - captures the img
        #.imshow - shows the image
        #.waitkey - adds a delay before the next capture
# https://prod.liveshare.vsengsaas.visualstudio.com/join?3191DFC56B03C3D14E973AF0FD04B073A8A
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

    cv2.imshow("img", img)
    
    fps = 1 / (c_time - p_time)

    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)},', (40, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)

    cv2.waitKey(1)

