import cv2
import numpy as np
from matplotlib import image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



# addressing the handlandmark model
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
model_options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2) #since we are using two hands that's why!
model_detector = vision.HandLandmarker.create_from_options(model_options)

# Testing with image file first
test_image = mp.Image.create_from_file('tri.jpg')

#testing with live video feed from webcam
live_feed = cv2.VideoCapture(1)
_, frames = live_feed.read()

# landmarks on video feed function
def draw_landmarks_on_video(video, result):
    annotated_video = video.copy()

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * video.shape[1])
                y = int(landmark.y * video.shape[0])
                cv2.circle(annotated_video, (x, y), 5, (0, 255, 0), -1)

    return annotated_video


#anotating function
# def draw_landmarks_on_image(image, result):
#     annotated_image = image.copy()
#     if result.hand_landmarks:
#         for hand_landmarks in result.hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 x = int(landmark.x * image.shape[1])
#                 y = int(landmark.y * image.shape[0])
#                 cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
#     return annotated_image

def draw_landmarks_on_image(image, result):
    annotated_image = image.copy()

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

    return annotated_image




# detection_result = model_detector.detect(test_image)
detection_result = model_detector.detect(frames)
# annotated_image = draw_landmarks_on_image(test_image.numpy_view(), detection_result)
annotated_video = draw_landmarks_on_video(frames, detection_result)
# cv2.imshow("New Image",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.imshow("Live Feed",cv2.cvtColor(annotated_video, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
Old Code [Not - Working]
"""



# import cv2, mediapipe as mp, asyncio, websockets, json, math

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2)

# cap = cv2.VideoCapture(0)

# def finger_extended(tip, pip):
#     return tip.y < pip.y

# def detect_fire_jutsu(hand):
#     lm = hand.landmark
#     index = finger_extended(lm[8], lm[6])
#     middle = finger_extended(lm[12], lm[10])
#     ring = finger_extended(lm[16], lm[14])
#     pinky = finger_extended(lm[20], lm[18])
#     return index and middle and not ring and not pinky

# async def handler(ws):
#     while True:
#         _, frame = cap.read()
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = hands.process(rgb)

#         event = "NONE"

#         if res.multi_hand_landmarks:
#             for hand in res.multi_hand_landmarks:
#                 if detect_fire_jutsu(hand):
#                     event = "FIRE_JUTSU"

#         await ws.send(json.dumps({"event": event}))
#         await asyncio.sleep(0.03)

# async def main():
#     async with websockets.serve(handler, "localhost", 8765):
#         await asyncio.Future()

# asyncio.run(main())