import cv2
face_classifier = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

cap = cv2.VideoCapture(0)

while True:

    ret, frames = cap.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    detect_face = face_classifier.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in detect_face:
        cv2.rectangle(frames, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face Detection", frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("quitting...")
        break

cap.release()
cv2.destroyAllWindows()