import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from collections import deque
import asyncio
import websockets
import json
import threading


# =========================
# MediaPipe Model
# =========================
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

model_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

model_detector = vision.HandLandmarker.create_from_options(model_options)


# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(1)
timestamp = 0


# =========================
# Hand connections
# =========================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


# =========================
# Gesture smoothing
# =========================
gesture_buffer = deque(maxlen=5)

def smooth_gesture(new_gesture):
    gesture_buffer.append(new_gesture)
    if gesture_buffer.count(new_gesture) >= 3:
        return new_gesture
    return "NONE"


# =========================
# Geometry helpers
# =========================
def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    return math.degrees(math.acos(dot / (mag1 * mag2 + 1e-6)))


def finger_up(hand, tip, pip):
    return hand[tip].y < hand[pip].y


def get_finger_states(hand):
    return {
        "index":  finger_up(hand, 8, 6),
        "middle": finger_up(hand, 12, 10),
        "ring":   finger_up(hand, 16, 14),
        "pinky":  finger_up(hand, 20, 18),
    }


def thumb_up(hand):
    wrist = hand[0]
    thumb_tip = hand[4]
    index_mcp = hand[5]

    thumb_vec = (thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
    index_vec = (index_mcp.x - wrist.x, index_mcp.y - wrist.y)

    return angle_between(thumb_vec, index_vec) > 40


def detect_fire_jutsu(hand):
    fingers = get_finger_states(hand)
    return (
        fingers["index"]
        and fingers["middle"]
        and not fingers["ring"]
        and not fingers["pinky"]
    )


# =========================
# Drawing
# =========================
def draw_landmarks_on_video(frame, result):
    annotated = frame.copy()
    h, w, _ = frame.shape

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            points = []
            for lm in hand:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

            for a, b in HAND_CONNECTIONS:
                cv2.line(annotated, points[a], points[b], (0, 255, 0), 2)

    return annotated


# =========================
# WebSocket Server
# =========================
clients = set()

async def ws_handler(websocket):
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        clients.remove(websocket)


async def broadcast(event):
    if clients:
        msg = json.dumps({"gesture": event})
        await asyncio.gather(*(c.send(msg) for c in clients))


async def ws_server():
    async with websockets.serve(ws_handler, "localhost", 8765):
        await asyncio.Future()


def start_ws_server():
    asyncio.run(ws_server())


# Run WebSocket server in background thread
threading.Thread(target=start_ws_server, daemon=True).start()


# =========================
# Main Loop
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = model_detector.detect_for_video(mp_image, timestamp)

    gesture = "NONE"

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            if detect_fire_jutsu(hand):
                gesture = "FIRE_JUTSU"

    stable = smooth_gesture(gesture)

    if stable != "NONE":
        asyncio.run(broadcast(stable))
        print("🔥 Gesture:", stable)

    annotated = draw_landmarks_on_video(frame, result)
    cv2.imshow("Live Feed", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    timestamp += 33


cap.release()
cv2.destroyAllWindows()
