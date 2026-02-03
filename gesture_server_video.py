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

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9,10), (10,11), (11,12),      # Middle
    (0,13), (13,14), (14,15), (15,16),     # Ring
    (0,17), (17,18), (18,19), (19,20)      # Pinky
]

def draw_landmarks_on_video(video, result):
    annotated_video = video.copy()
    # HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


    h, w, _ = video.shape

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Convert landmarks to pixel coords once
            points = []
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                cv2.circle(annotated_video, (x, y), 4, (0, 255, 0), -1)

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                pt1 = points[start_idx]
                pt2 = points[end_idx]
                cv2.line(annotated_video, pt1, pt2, 0xfff, 2)

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