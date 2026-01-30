import cv2


image = cv2.imread('pop_kaal.jpg', 1)

#if image shape
# print(image.shape)

if image is not None:
    # resizedImage = cv2.resize(image, (400,400))
    # cropped = image[300:900, 500:1040]
    # h, w, c = image.shape
    h = image.shape[0]
    w = image.shape[1]
    center = (w//2, h//2)
    radius = 1000
    text = "hello bhai"
    # print(h, w)
    # print(f"Height: {h}\nWidth: {w}\nColor Channel: {c}")
    # m = cv2.getRotationMatrix2D(center,90,1.0)
    # rotatedImage = cv2.warpAffine(image, m, (w,h))
    # flippedImage = cv2.flip(rotatedImage, -1)
    # modifiedImage = cv2.line(image, (100,50), (500,50), 0xfff, 5)
    # modifiedImage = cv2.rectangle(image, (100,50), (500,100), 0xfff, 5)
    modifiedImage = cv2.circle(image, center, radius, 0xfff, 5)
    modifiedImage = cv2.putText(image, text, (200,300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0xfff, 5)
    cv2.imshow("original image", image)
    cv2.imshow("new image", modifiedImage)
    # cv2.imshow("rotated image", rotatedImage)
    # cv2.imshow("flipped image", flippedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image save failed.")

