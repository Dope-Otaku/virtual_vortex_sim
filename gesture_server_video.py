import cv2
import numpy as np
from matplotlib import image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision




# addressing the handlandmark model
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
model_options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO) #since we are using two hands that's why!
model_detector = vision.HandLandmarker.create_from_options(model_options)

cap = cv2.VideoCapture(1)

timestamp = 0

def draw_landmarks_on_video(video, result):
    annotated_video = video.copy()

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * video.shape[1])
                y = int(landmark.y * video.shape[0])
                cv2.circle(annotated_video, (x, y), 5, (0, 255, 0), -1)

    return annotated_video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    detection_result = model_detector.detect_for_video(
        mp_image,
        timestamp
    )

    annotated = draw_landmarks_on_video(frame, detection_result)

    cv2.imshow("Live Feed", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    timestamp += 33  # ~30 FPS

cap.release()
cv2.destroyAllWindows()