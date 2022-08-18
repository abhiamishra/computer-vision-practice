import cv2
import mediapipe as mp
import time
import handtrackmodule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)

detector = htm.handDetector(detectionCon=0.75)

    #To run our webcam
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX,3,(255,8,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)