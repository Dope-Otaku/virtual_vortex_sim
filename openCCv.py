import cv2


image = cv2.imread('pop_kaal.jpg', 1)

if image is not None:
    success =  cv2.imwrite('hello.jpg', image)
    print("Image saved succesfully")
else:
    print("Image save failed.")

