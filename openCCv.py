import cv2
import numpy as np


image = cv2.imread("pop_kaal.jpg")
sharpened_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharepened_image = cv2.filter2D(image, -1, sharpened_kernel)

if image is None:
    print("no image found")

cv2.imshow("original Image", image)
cv2.imshow("blurred Image", sharepened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
