import cv2
import numpy as np
import time
import pyautogui
import HandTracking as htm

# ================== CONFIG ==================
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
# ===========================================

pTime = 0
plocX, plocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()

while True:
    success, img = cap.read()
    if not success:
        continue

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    fingers = detector.fingersUp()
    if len(fingers) != 5:
        cv2.imshow("Virtual Mouse", img)
        cv2.waitKey(1)
        continue

    cv2.rectangle(
        img,
        (frameR, frameR),
        (wCam - frameR, hCam - frameR),
        (255, 0, 255),
        2
    )

    # ========== MOVE ==========
    if fingers[1] == 1 and fingers[2] == 0:
        x1, y1 = lmList[8][1:]
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        pyautogui.moveTo(wScr - clocX, clocY)
        plocX, plocY = clocX, clocY

    # ========== CLICK ==========
    if fingers[1] == 1 and fingers[2] == 1:
        length, img, _ = detector.findDistance(8, 12, img)
        if length and length < 40:
            pyautogui.click()
            time.sleep(0.25)

    # ========== FPS ==========
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime else 0
    pTime = cTime

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (20, 50),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2
    )

    cv2.imshow("Virtual Mouse", img)
    cv2.waitKey(1)
