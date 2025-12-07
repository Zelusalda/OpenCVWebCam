import cv2

webCam = cv2.VideoCapture(0)
classificadorFace = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

while True:
    camera, frame = webCam.read()
    faceGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classificadorFace.detectMultiScale(faceGray, scaleFactor= 1.1, minNeighbors= 6)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        localEyes = frame[y:y + h, x:x + w]
        eyesGray = cv2.cvtColor(localEyes, cv2.COLOR_BGR2GRAY)

        eyes = classificadorOlhos.detectMultiScale(eyesGray, scaleFactor=1.08, minNeighbors=6)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(localEyes, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
    cv2.imshow("WebCam", frame)

    if cv2.waitKey(1) == ord('q'):
        break

webCam.release()
cv2.destroyAllWindows()
