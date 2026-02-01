import cv2
import numpy as np

x= 300
y= 300

xCenter = int(x//2)
yCenter = int(y//2)

img1 = np.zeros((x,y), dtype="uint8")
img2 = np.zeros((x,y), dtype="uint8")

circle = cv2.circle(img1, (xCenter, yCenter), (xCenter-50), 255, -1)
rectangle = cv2.rectangle(img2, (100, 100), (250, 250),255,-1)


bitwise_and = cv2.bitwise_and(img1, img2)
bitwise_or = cv2.bitwise_or(img1, img2)
bitwise_not = cv2.bitwise_not(img1)

cv2.imshow("Circle", circle)
cv2.imshow("rectangle", rectangle)
cv2.imshow("bitwise_and", bitwise_and)
cv2.imshow("bitwise_or", bitwise_or)
cv2.imshow("bitwise_not", bitwise_not)
cv2.waitKey(0)
cv2.destroyAllWindows()
