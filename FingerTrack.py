import cv2
import mediapipe as mp
import pyautogui
import screeninfo

screen = screeninfo.get_monitors()[0]
width, height = screen.width, screen.height

def move_mouse(x, y):
    tx = int(x * width)
    ty = int(y * height)
    pyautogui.moveTo(tx, ty)

def track_finger():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        img = cv2.flip(img, 1)

        results = hands.process(img)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                i_l = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = i_l.x, i_l.y
                move_mouse(x, y)

        cv2.imshow('Finger Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_finger()
