import cv2
import math
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpdraw = mp.solutions.drawing_utils


def calculate_hand_distance(h1_l, h2_l):
    tip1 = h1_l[8]
    tip2 = h2_l[8]
    x1, y1 = tip1[1], tip1[2]
    x2, y2 = tip2[1], tip2[2]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 480)
mylmList = []
img_counter = 0

while True:
    isopen, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    allHands = []
    h, w, c = frame.shape

    if results.multi_hand_landmarks:
        for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
            myHand = {}
            mylmList = []
            xList = []
            yList = []

            for id, lm in enumerate(handLms.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                mylmList.append([id, px, py])
                # print(f'{id} : {px},{py}')
                xList.append(px)
                yList.append(py)

            myHand["lmList"] = mylmList
            myHand["type"] = handType.classification[0].label
            allHands.append(myHand)

            mpdraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            # cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),3)

    if len(allHands) >= 2:
        h1_l = allHands[0]["lmList"]
        h2_l = allHands[1]["lmList"]
        dist = calculate_hand_distance(h1_l, h2_l)
        print(int(dist))
        tip1 = h1_l[8]
        tip2 = h2_l[8]
        x1, y1 = tip1[1], tip1[2]
        x2, y2 = tip2[1], tip2[2]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
    cv2.imshow("Video Capture", frame)

    k = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
