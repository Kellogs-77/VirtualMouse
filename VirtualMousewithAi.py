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
        x3, y3 = lmList[20][1:]
        x4, y4 = lmList[4][1:]

        # print(x1, y1, x2, y2)

        # 3. Check which finger are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(
            img, (frameR, frameR), (640 - frameR, 480 - frameR), (167, 0, 180), 2
        )
        # 4. Only index Finger : Moving Mode
        if (
            fingers[0] == 0
            and fingers[1] == 1
            and fingers[2] == 0
            and fingers[3] == 0
            and fingers[4] == 0
        ):
            # 5. Convert Coordinates
            x5 = np.interp(x1, (frameR, 640 - frameR), (0, wScr))
            y5 = np.interp(y1, (frameR, 480 - frameR), (0, hScr))

            # 6. Smoothen Values
            currLocX = prevLocX + (x5 - prevLocX) / smoothening
            currLocY = prevLocY + (y5 - prevLocY) / smoothening

            # 7. Move mouse
            autopy.mouse.move(wScr - currLocX, currLocY)
            cv2.circle(img, (x1, y1), 15, (167, 0, 180), cv2.FILLED)
            prevLocX, prevLocY = currLocX, currLocY

        # 8. Both index and middle fingers are up: Clicking Mode
        if (
            fingers[0] == 0
            and fingers[1] == 1
            and fingers[2] == 1
            and fingers[3] == 0
            and fingers[4] == 0
        ):

            # 9. Find distance btween fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # 10. Click mouse if distance short
            if length < 70:
                cv2.circle(
                    img, (lineInfo[4], lineInfo[5]), 15, (255, 255, 255), cv2.FILLED
                )
                pyautogui.click()

        # 11. Both index and pinky fingers are up: Scrolling Mode - Up
        if (
            fingers[0] == 0
            and fingers[1] == 1
            and fingers[2] == 0
            and fingers[3] == 0
            and fingers[4] == 1
        ):
            cv2.circle(img, (x1, y1), 15, (167, 0, 180), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (167, 0, 180), cv2.FILLED)
            pyautogui.scroll(5)

        # 12. Only pinky finger is up: Scrolling Mode - Down
        if (
            fingers[0] == 0
            and fingers[1] == 0
            and fingers[2] == 0
            and fingers[3] == 0
            and fingers[4] == 1
        ):
            cv2.circle(img, (x3, y3), 15, (167, 0, 180), cv2.FILLED)
            pyautogui.scroll(-5)
        # 13. Only thumb is up: Double Click Mode
        if (
            fingers[0] == 1
            and fingers[1] == 0
            and fingers[2] == 0
            and fingers[3] == 0
            and fingers[4] == 0
        ):
            cv2.circle(img, (x4, y4), 15, (167, 0, 180), cv2.FILLED)
            pyautogui.doubleClick()
        # middle, ring, pinky up and thumb and pointer down: Close the program
        if (
            fingers[0] == 0
            and fingers[1] == 0
            and fingers[2] == 1
            and fingers[3] == 1
            and fingers[4] == 1
        ):
            break
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
