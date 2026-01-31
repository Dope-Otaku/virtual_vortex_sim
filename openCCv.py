import cv2


image = cv2.imread("pop_kaal.jpg")
blurred_image = cv2.GaussianBlur(image, (7,7), 11)

if image is None:
    print("no image found")

cv2.imshow("original Image", image)
cv2.imshow("blurred Image", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
