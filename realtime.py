# realtime.py -- MediaPipe hand gestures for Subway Surfers
import cv2
import mediapipe as mp
import pyautogui
import time

last_action_time = 0
cooldown = 0.1  # 100 ms

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Gesture state
last_gesture = None

# Map gestures to actions
gesture_actions = {
    "THUMB": "left",
    "PINKY": "right",
    "INDEX": "up",      # jump
    "PEACE": "down",    # roll
}

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "FIST"  # default if nothing detected

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark

            # Simple finger state detection
            def finger_extended(tip_id, pip_id):
                return lm[tip_id].y < lm[pip_id].y  # tip above pip in image coords

            thumb = lm[4].x < lm[3].x  # for right hand (mirror image)
            index = finger_extended(8, 6)
            middle = finger_extended(12, 10)
            ring = finger_extended(16, 14)
            pinky = finger_extended(20, 18)

            if thumb and not (index or middle or ring or pinky):
                gesture = "THUMB"
            elif pinky and not (thumb or index or middle or ring):
                gesture = "PINKY"
            elif index and not (middle or ring or pinky or thumb):
                gesture = "INDEX"
            elif index and middle and not (ring or pinky or thumb):
                gesture = "PEACE"

            # Reset rule: fist clears the gesture so it can be repeated
            if gesture == "FIST":
                last_gesture = None

        # Action triggering logic with cooldown
        current_time = time.time()
        if gesture != last_gesture and (current_time - last_action_time) > cooldown:
            if gesture in gesture_actions:
                pyautogui.press(gesture_actions[gesture])

            last_gesture = gesture
            last_action_time = current_time

        cv2.putText(frame, gesture, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Gesture Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
