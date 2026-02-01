import cv2
import numpy as np

image = cv2.imread("tri.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)


contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("countours failed to run")

cv2.drawContours(image, contours, -1, 0xfff, 3)
    
# image = cv2.imshow("original image", image)
modified_image = cv2.imshow("Modified Image", image)


cv2.waitKey(0)
cv2.destroyAllWindows()
