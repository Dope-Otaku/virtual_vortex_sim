import cv2


video = cv2.VideoCapture(1)

#if image shape
# print(image.shape)

while True:
    ret, frame = video.read()
    # print("press q to quit the window.....") don't use it here it will flood the console

    if not ret:
        print("could not read frame")
        break

    cv2.imshow("Video Capture", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        print("quitting...")
        break

video.release()
cv2.destroyAllWindows()

