import cv2

webCam = cv2.VideoCapture(0)

while True:
    camera, frame = webCam.read()

    cv2.imshow("Imagem da WebCam", frame)

    if cv2.waitKey(1) == ord('q'):
        break
webCam.release()
cv2.destroyAllWindows()