# realtime.py -- simple MediaPipe + rule-based gestures
import cv2
import mediapipe as mp
import numpy as np
# import pyautogui   # uncomment when you want to send keys

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)   # change 0 -> 1 if your camera is different

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't read camera. Exiting.")
            break

        frame = cv2.flip(frame, 1)               # mirror view
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "No hand"
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            # normalized coords (0..1), easy to use across distances
            index_x = lm[8].x
            wrist = lm[0]

            # left / right by index finger x position
            if index_x < 0.40:
                gesture = "LEFT"
            elif index_x > 0.60:
                gesture = "RIGHT"
            else:
                # palm vs fist: average distance of tips from wrist
                tips = [8, 12, 16, 20, 4]  # index,middle,ring,pinky,thumb
                dists = [np.linalg.norm(np.array([lm[t].x - wrist.x, lm[t].y - wrist.y])) for t in tips]
                avg = np.mean(dists)
                # threshold chosen empirically for normalized coords
                gesture = "PALM" if avg > 0.22 else "FIST"

            # Example: map gesture to key (uncomment to enable)
            # if gesture == "LEFT":
            #     pyautogui.press('left')
            # elif gesture == "RIGHT":
            #     pyautogui.press('right')
            # elif gesture == "PALM":
            #     pyautogui.press('up')
            # elif gesture == "FIST":
            #     pyautogui.press('down')

        cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Hand Gesture Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
