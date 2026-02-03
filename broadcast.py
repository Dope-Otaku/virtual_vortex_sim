import asyncio
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import websockets

# ---------- MediaPipe ----------
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp = 0

gesture_buffer = deque(maxlen=5)

# ---------- Gesture helpers ----------
def finger_up(hand, tip, pip):
    return hand[tip].y < hand[pip].y

def thumb_up(hand):
    return hand[4].x > hand[3].x  # works for right hand

def detect_gesture(hand):
    if thumb_up(hand):
        return "THUMBS_UP"
    if finger_up(hand, 8, 6):
        return "INDEX_UP"
    return "NONE"

def smooth(gesture):
    gesture_buffer.append(gesture)
    if gesture_buffer.count(gesture) >= 3:
        return gesture
    return "NONE"

# ---------- WebSocket handler ----------
async def handler(ws):
    global timestamp
    print("🟢 Frontend connected")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect_for_video(mp_image, timestamp)
            event = "NONE"

            if result.hand_landmarks:
                event = detect_gesture(result.hand_landmarks[0])
                event = smooth(event)

            await ws.send(json.dumps({"event": event}))
            await asyncio.sleep(0.03)
            timestamp += 33

    except websockets.exceptions.ConnectionClosed:
        print("🔴 Frontend disconnected")

# ---------- Run server ----------
async def main():
    async with websockets.serve(handler, "127.0.0.1", 8765):
        print("🚀 Gesture server running on ws://127.0.0.1:8765")
        await asyncio.Future()

asyncio.run(main())
