import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from mediapipe import solutions

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    #pose_landmarks_list = detection_result
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

my_result = 0
video = cv2.VideoCapture(0)

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    my_result = result
    print('gesture recognition result: {}'.format(result))


options = GestureRecognizerOptions(base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM)

timestamp = 0
with GestureRecognizer.create_from_options(options) as recognizer:
  # The recognizer is initialized. Use it here.
    while video.isOpened(): 
        # Capture frame-by-frame
        ret, frame = video.read()

        if not ret:
            print("Ignoring empty frame")
            break

        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Send live image data to perform gesture recognition
        # The results are accessible via the `result_callback` provided in
        # the `GestureRecognizerOptions` object.
        # The gesture recognizer must be created with the live stream mode.
        my_result = recognizer.recognize_async(mp_image, timestamp)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), my_result)
        cv2.imshow('Show',annotated_image)  
        if cv2.waitKey(5) & 0xFF == 'q':
            break

video.release()
cv2.destroyAllWindows()