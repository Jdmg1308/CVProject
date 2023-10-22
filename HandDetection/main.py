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
    num = int(0)

    hands = mp_hands.Hands()
    while True:
        data, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            index = results.multi_hand_landmarks[0].landmark[8]

            x = int(index.x * width)
            y = int(index.y * height)
            
            cv2.circle(img=image, center=(x,y), radius = 10, color=(100,100,100))
            index_x = screen_width/width * x
            index_y = screen_height/height * y
            pyautogui.moveTo(index_x, index_y)

        # gestures will resturn result
        #if result == no_gesture) == move_mouse

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

    ratio = screen_height/screen_width

    window_width = 1080
    window_height = ratio * window_width

    capture_hand(window_width, window_height, mp_hands)



if __name__ == "__main__":
    gen_hand()
    