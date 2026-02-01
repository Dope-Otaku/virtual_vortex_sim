import cv2
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_classifier = cv2.CascadeClassifier("haarcascade_smile.xml")

"""
currently the faces and smile contors are not alligned

"""

cap = cv2.VideoCapture(0)

while True:

    ret, frames = cap.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    detect_face = face_classifier.detectMultiScale(gray, 1.1, 10)
    
    

    for (x, y, w, h) in detect_face:
        cv2.rectangle(frames, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frames, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frames[y:y+h, x:x+w]

        detect_eyes = eyes_classifier.detectMultiScale(roi_gray, 1.1, 10)

        if len(detect_eyes)>0:
            for (x, y, w, h) in detect_eyes:
                cv2.rectangle(frames, (x,y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(frames, "Eyes", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        detect_smile = smile_classifier.detectMultiScale(roi_gray, 1.7, 20)

        if len(detect_smile)>0:
            for (x, y, w, h) in detect_smile:
                cv2.rectangle(frames, (x,y), (x+w, y+h), (0,0,255), 2)
                cv2.putText(frames, "Smile", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("Face Detection", frames)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("quitting...")
        break

cap.release()
cv2.destroyAllWindows()