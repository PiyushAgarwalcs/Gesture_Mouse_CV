import cv2
import mediapipe as mp
import math
import os

# ================== MODEL PATH ==================
MODEL_PATH = r"D:\Gesture_Mouse\hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

# ================== MediaPipe Tasks ==================
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


class handDetector:
    def __init__(self, maxHands=1):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=maxHands
        )
        self.detector = HandLandmarker.create_from_options(options)
        self.lmList = []
        self.results = None
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imgRGB
        )

        self.results = self.detector.detect(mp_image)

        if draw and self.results.hand_landmarks:
            h, w, _ = img.shape
            for hand in self.results.hand_landmarks:
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return img

    def findPosition(self, img, handNo=0):
        self.lmList = []

        if not self.results or not self.results.hand_landmarks:
            return self.lmList, []

        hand = self.results.hand_landmarks[handNo]
        h, w, _ = img.shape

        xList, yList = [], []

        for id, lm in enumerate(hand):
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            self.lmList.append([id, cx, cy])

        bbox = [min(xList), min(yList), max(xList), max(yList)]
        return self.lmList, bbox

    def fingersUp(self):
        if len(self.lmList) != 21:
            return []

        fingers = []

        # Thumb (right hand)
        fingers.append(
            1 if self.lmList[self.tipIds[0]][1] >
                 self.lmList[self.tipIds[0] - 1][1] else 0
        )

        # Other fingers
        for i in range(1, 5):
            fingers.append(
                1 if self.lmList[self.tipIds[i]][2] <
                     self.lmList[self.tipIds[i] - 2][2] else 0
            )

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        if len(self.lmList) != 21:
            return None, img, []

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
