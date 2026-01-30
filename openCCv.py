import cv2


image = cv2.imread('pop_kaal.jpg', 1)

if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully.")