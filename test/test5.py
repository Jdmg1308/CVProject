#@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# import urllib

# IMAGE_FILENAMES = ['thumbs_down.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}'
#   urllib.request.urlretrieve(url, name)

# plt.rcParams.update({
#     'axes.spines.top': False,
#     'axes.spines.right': False,
#     'axes.spines.left': False,
#     'axes.spines.bottom': False,
#     'xtick.labelbottom': False,
#     'xtick.bottom': False,
#     'ytick.labelleft': False,
#     'ytick.left': False,
#     'xtick.labeltop': False,
#     'xtick.top': False,
#     'ytick.labelright': False,
#     'ytick.right': False
# })

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

import cv2
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image, window_name="Image"):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow(window_name, img)


# Preview the images.
# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# display_batch_of_images_with_gestures_and_hand_landmarks(images, results)

import cv2
import mediapipe as mp
import keyboard

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

# Set the desired window size
window_width = 1080  # Change this to your desired width
window_height = 720  # Change this to your desired height

# OpenCV window settings
#cv2.resizeWindow("Handtracker", window_width, window_height)

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

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
                cv2.circle(image, (x, y), 6, (12, 124, 194), -1)  # white color (BGR format)

            for connection in mphands.HAND_CONNECTIONS:
                x0, y0 = int(hand_landmarks.landmark[connection[0]].x * image.shape[1]), int(hand_landmarks.landmark[connection[0]].y * image.shape[0])
                x1, y1 = int(hand_landmarks.landmark[connection[1]].x * image.shape[1]), int(hand_landmarks.landmark[connection[1]].y * image.shape[0])
                cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), 4)  # black color (BGR format)

                cv2.line(image, (x0, y0), (x1, y1), (255, 255, 255), 2)  # white color (BGR format)

    #cv2.namedWindow("Handtracker", cv2.WINDOW_NORMAL)
    # STEP 3: Load the input image.
    # image = mp.Image.create_from_file(image_file_name)

    # STEP 4: Recognize gestures in the input image.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    recognition_result = recognizer.recognize(mp_image)

    # STEP 5: Process the result. In this case, visualize it.
    top_gesture = recognition_result.gestures
    if recognition_result.gestures != []:
        print(top_gesture)
    hand_landmarks = recognition_result.hand_landmarks
    # results.append((top_gesture, hand_landmarks))

    cv2.imshow("Handtracker", image)
    #cv2.resizeWindow("Handtracker", window_width, window_height)
    cv2.waitKey(1)

    if(keyboard.is_pressed('q')):
        break

cap.release()

cv2.destroyAllWindows()