import cv2
import mediapipe as mp
import keyboard
import pyautogui

pointer_finger = int(8)
thumb = int(4)
index_y = 0
thumb_y = 0
index_x = 0
screen_width, screen_height = pyautogui.size()


def move_mouse(current_img, land_marks, width, height):
    for land_mark in enumerate(land_marks):
        x = int(land_mark.x * width)
        y = int(land_mark.y * height)

        if land_mark == pointer_finger:
            cv2.circle(img=current_img, center=(x,y), radius = 10, color=(100,100,100))
            index_x = screen_width/width * x
            index_y = screen_height/height * y
            pyautogui.moveTo(index_x, index_y)

        if land_mark == thumb:
            cv2.circle(img=current_img, center=(x,y), radius = 10, color=(100,100,100))
            thumb_y = screen_height/height * y
    ##print("life is roblox")

def mouse_click(position_x, position_y):
    #click at x and y coordinates
    pyautogui.click(position_x, position_y)
    pyautogui.sleep(1)


def hand_options():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return mp.solutions.hands

def capture_hand(width, height, mp_hands):
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    hands = mp_hands.Hands()

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
                    cv2.circle(image, (x, y), 6, (12, 124, 194), -1)  # white color (BGR format)

                for connection in mp_hands.HAND_CONNECTIONS:
                    x0, y0 = int(hand_landmarks.landmark[connection[0]].x * image.shape[1]), int(hand_landmarks.landmark[connection[0]].y * image.shape[0])
                    x1, y1 = int(hand_landmarks.landmark[connection[1]].x * image.shape[1]), int(hand_landmarks.landmark[connection[1]].y * image.shape[0])
                    cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), 4)  # black color (BGR format)

                    cv2.line(image, (x0, y0), (x1, y1), (255, 255, 255), 2)  # white color (BGR format)

        # gestures will resturn result
        #if result == no_gesture) == move_mouse
        move_mouse(image, results.multi_hand_landmarks, width, height)

        #if result == point up
        # if abs(thumb_y - index_y < 20):
        #         mouse_click(index_x, index_y)

        cv2.imshow("Handtracker", image)
        cv2.waitKey(1)

        if(keyboard.is_pressed('q')):
            break

    cap.release()

    cv2.destroyAllWindows()

def gen_hand():
    mp_hands = hand_options()

    window_width = 1080
    window_height = 720

    capture_hand(window_width, window_height, mp_hands)



if __name__ == "__main__":
    gen_hand()
    