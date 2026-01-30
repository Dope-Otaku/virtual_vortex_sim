import cv2


image = cv2.imread('pop_kaal.jpg', 1)

#if image shape
# print(image.shape)

if image is not None:
    # resizedImage = cv2.resize(image, (400,400))
    cropped = image[300:900, 500:1040]
    h, w, c = cropped.shape
    print(f"Height: {h}\nWidth: {w}\nColor Channel: {c}")
    cv2.imshow("original image", image)
    cv2.imshow("cropped image", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image save failed.")

