import cv2
import mediapipe as mp
import osascript
from math import sqrt
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=.5, trackCon=.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        vol_max = img.shape[0]//2
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
                if id == 4:
                    vol_s = (cx, cy)
                if id == 8:
                    vol_e = (cx, cy)

            vol_bar_len = sqrt((vol_e[1] - vol_s[1])**2 + (vol_e[0] - vol_s[0])**2)
            cv2.line(img, vol_s, vol_e, (0, 0, 255), 2)
            target_volume = min(100*vol_bar_len//vol_max, 100)
            cv2.putText(img, "{}".format(target_volume), (40, 640), cv2.FONT_HERSHEY_SIMPLEX, 1, 25)
            cv2.line(img, (40, 600), (40, 600-3*int(target_volume)), (0, 0, 255), 20)
            vol = "set volume output volume " + str(target_volume)
            # osascript.osascript(vol)

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0) # For mac device the camera id is 1.
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()