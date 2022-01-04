import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui

frameR = 100  # Frame Reduction
webcam = cv2.VideoCapture(0)
smoothening = 5
prevTime = 0
prevLocX, prevLocY = 0, 0
currLocX, currLocY = 0, 0
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)


while True:
    # 1. Find hand landmarks
    success, img = webcam.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print(x1, y1, x2, y2)

        # 3. Check which finger are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(
            img, (frameR, frameR), (640 - frameR, 480 - frameR), (167, 0, 180), 2
        )
        # 4. Only index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, 640 - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, 480 - frameR), (0, hScr))

            # 6. Smoothen Values
            currLocX = prevLocX + (x3 - prevLocX) / smoothening
            currLocY = prevLocY + (y3 - prevLocY) / smoothening

            # 7. Move mouse
            autopy.mouse.move(wScr - currLocX, currLocY)
            cv2.circle(img, (x1, y1), 15, (167, 0, 180), cv2.FILLED)
            prevLocX, prevLocY = currLocX, currLocY
        # 8. Both index and middle fingers are up: Cliking Mode
        if fingers[1] == 1 and fingers[2] == 1:

            # 9. Find distance btween fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # 10. Click mouse if distance short
            if length < 70:
                cv2.circle(
                    img, (lineInfo[4], lineInfo[5]), 15, (255, 255, 255), cv2.FILLED
                )
                autopy.mouse.click()

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    # the '1' is the auto wait before the program hits a key so the webcame image can move to the next frame-otherwise it gets stuck until we manually press a key

    if key == 81 or key == 113:
        break


print("Code completed!")
