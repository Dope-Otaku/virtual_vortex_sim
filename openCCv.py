import cv2


image = cv2.imread('pop_kaal.jpg', 1)

#if image shape
# print(image.shape)

if image is not None:
    h, w, c = image.shape
    print(f"Height: {h}\nWidth: {w}\nColor Channel: {c}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale version", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image save failed.")

