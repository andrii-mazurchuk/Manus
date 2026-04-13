import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Try index 1 or 2.")
    exit(1)

print("Webcam OK. Press Q to quit.")

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                coords = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                label = result.multi_handedness[i].classification[0].label
                print(f"Hand {i+1} ({label}) — Wrist: {coords[0]}  Index tip: {coords[8]}  Thumb tip: {coords[4]}")

        cv2.imshow("Manus — Webcam Test (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Done.")
