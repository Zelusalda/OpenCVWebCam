import cv2

webCam = cv2.VideoCapture(0)
classificadorFace = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    camera, frame = webCam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detec = classificadorFace.detectMultiScale(gray)

    for(x, y, w, h) in detec:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("WebCam", frame)

        if cv2.waitKey(1) == ord('q'):
            break

webCam.release()
cv2.destroyAllWindowns()
