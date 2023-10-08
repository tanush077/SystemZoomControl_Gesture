import cv2
import mediapipe as mp
import pyautogui
import screeninfo

wCam,hCam = 1280,720

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

def move_mouse(x, y, width, height):
    tx = int(x * width)
    ty = int(y * height)
    invx = width - tx
    pyautogui.moveTo(invx, ty)

def track_finger():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    screen = screeninfo.get_monitors()[0]
    width, height = screen.width, screen.height

    while True:
        success, img = cap.read()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_l = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = index_l.x, index_l.y
                move_mouse(x, y, width, height)

        cv2.imshow('Finger Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_finger()
