import cv2 
import numpy as np
import time
import os
import math

import handtrackmodule as htm

brushThickness = 15
eraserThickness = 80

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
header = overlayList[0]

drawColor = (255, 253, 1)

#print(header)
cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 800)

detector = htm.handDetector(detectionCon=0.70)
xp, yp = 0, 0

imgCanvas = np.zeros((480, 848, 3), np.uint8)

while True:
    #Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        #print(lmList[8])
        #index finger = x1, y1
        x1, y1 = lmList[8][1:]

        #middle finger = x2, y2
        x2, y2 = lmList[12][1:]


        #Finding thumb and index finger
        x3, y3 = lmList[4][1], lmList[4][2]

        #Finding length
        length = math.hypot(x3-x1, y3-y1)

        #Check which fingers are up
        fingersUp = detector.fingersUp()
        #print(fingersUp)

        #Checking if font mode is on
        if fingersUp[1] and fingersUp[4]:
            #Finding midpoint
            cx, cy = (x1+x3)//2, (y1+y3)//2

            cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x3,y3), 10, (255,0,255), cv2.FILLED)
            cv2.line(img, (x1,y1), (x3,y3), (255,0,255), 3)
            cv2.circle(img, (cx,cy), 10, (0, 255, 0), cv2.FILLED)
            
            brushThickness = int(np.interp(length, [20,190], [8, 80]))
            eraserThickness = brushThickness

        #If selection mode - two fingers are up
        if fingersUp[1] and fingersUp[2]:
            xp, yp = 0, 0

            #print("selection")
            if y1 < 116:
                if 178<x1<260:
                    header = overlayList[0]
                    drawColor = (255, 253, 1)
                elif 340 < x1 < 420:
                    header = overlayList[1]
                    drawColor = (0, 128, 0)
                elif 490 < x1 < 550:
                    header = overlayList[2]
                    drawColor = (255, 255, 255)
                elif 620 < x1 < 720 :
                    header = overlayList[3]
                    drawColor = (0,0,0)
            
            
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawColor, cv2.FILLED)


            #Checking for the click
            # we are in the header

        #If draw mode - index finger is up
        if fingersUp[1] and fingersUp[2]==False and fingersUp[4]==False:

            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            #print("draw")
            if xp==0 and yp==0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp),(x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp,yp),(x1,y1), drawColor, int(brushThickness))
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColor, int(brushThickness))

            xp, yp = x1, y1

    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    print(img.shape)
    print(imgCanvas.shape)

    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    #Setting the header image
    img[10:116, 30:796] = header

    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    #cv2.imshow("ImageCanvs", imgCanvas)
    cv2.waitKey(1)