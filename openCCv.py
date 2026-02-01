import cv2
import numpy as np


image = cv2.imread("pop_kaal.jpg", cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(image, 50, 150)

cv2.imshow("original Image", image)
cv2.imshow("blurred Image", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
