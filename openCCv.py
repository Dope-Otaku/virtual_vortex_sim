import cv2


video = cv2.VideoCapture(1)

frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

codec = cv2.VideoWriter_fourcc(*'mp4v')
recordedFootage = cv2.VideoWriter("my_video.mp4", codec, 30, (frame_width,frame_height))

#if image shape
# print(image.shape)

while True:
    ret, frame = video.read()
    # print("press q to quit the window.....") don't use it here it will flood the console

    if not ret:
        print("could not read frame")
        break

    recordedFootage.write(frame)
    cv2.imshow("Recording Live", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        print("quitting...")
        break

video.release()
recordedFootage.release()
cv2.destroyAllWindows()

