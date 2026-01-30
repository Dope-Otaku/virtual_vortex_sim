import cv2


image = cv2.imread('pop_kaal.jpg', 1)

#if image shape
# print(image.shape)

if image is not None:
    resizedImage = cv2.resize(image, (400,400))
    h, w, c = resizedImage.shape
    print(f"Height: {h}\nWidth: {w}\nColor Channel: {c}")
    cv2.imshow("resized image", resizedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image save failed.")

