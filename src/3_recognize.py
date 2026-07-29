import cv2
import pickle
import numpy as np
import mediapipe as mp
import time
import os
import sys
sys.path.append(os.path.dirname(__file__))
from collections import deque, Counter
from utils import extract_landmarks

MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'gesture_model.pkl')

with open(MODEL_FILE, 'rb') as f:
    data = pickle.load(f)
model = data['model']
le    = data['encoder']

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

BUFFER_SIZE    = 12
CONFIDENCE_MIN = 0.50   # lowered from 0.75
HOLD_SECONDS   = 1.5

pred_buffer  = deque(maxlen=BUFFER_SIZE)
sentence     = []
last_gesture = ""
last_time    = time.time()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Recognition started!")
print("Q: quit | C: clear sentence | SPACE: add space")

with mp_hands.Hands(
    min_detection_confidence=0.5,   # lowered from 0.75
    min_tracking_confidence=0.5     # lowered from 0.75
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        coords, hand_lms = extract_landmarks(frame, hands)
        stable_gesture   = ""

        # Always show hand detection status
        if coords is not None:
            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,120), thickness=2),
                mp_draw.DrawingSpec(color=(255,0,200), thickness=2)
            )

            probs      = model.predict_proba([coords])[0]
            top_idx    = np.argmax(probs)
            confidence = probs[top_idx]

            # Always show what is being detected even below threshold
            raw_gesture = le.classes_[top_idx]
            cv2.putText(frame, f"Detecting: {raw_gesture} ({confidence*100:.0f}%)",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)

            if confidence >= CONFIDENCE_MIN:
                pred_buffer.append(raw_gesture)
            else:
                pred_buffer.append("")

            if pred_buffer:
                most_common, votes = Counter(pred_buffer).most_common(1)[0]
                if votes >= BUFFER_SIZE * 0.6 and most_common:
                    stable_gesture = most_common

                    cv2.putText(frame, f"{stable_gesture}",
                        (30, 110), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0,255,100), 3)
                    cv2.putText(frame, f"{confidence*100:.0f}% confidence",
                        (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,255,180), 2)

                    if stable_gesture != last_gesture:
                        last_gesture = stable_gesture
                        last_time    = time.time()
                    else:
                        elapsed  = time.time() - last_time
                        progress = min(elapsed / HOLD_SECONDS, 1.0)

                        bar_x = w // 2 - 160
                        bar_w = int(320 * progress)
                        cv2.rectangle(frame, (bar_x, h-120), (bar_x+320, h-96), (40,40,40), -1)
                        cv2.rectangle(frame, (bar_x, h-120), (bar_x+bar_w, h-96), (0,200,255), -1)
                        cv2.putText(frame, "Hold to add...",
                            (bar_x, h-128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                        if elapsed >= HOLD_SECONDS:
                            if not sentence or sentence[-1] != stable_gesture:
                                sentence.append(stable_gesture)
                            last_time = time.time() + 1.0
        else:
            # Show clearly that no hand is detected
            cv2.putText(frame, "NO HAND DETECTED",
                (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,255), 3)
            pred_buffer.append("")

        cv2.rectangle(frame, (0, h-88), (w, h), (20,20,20), -1)
        display = "  ".join(sentence[-14:]) if sentence else "(no gestures yet)"
        cv2.putText(frame, f"Sentence: {display}",
            (20, h-54), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
        cv2.putText(frame, "Q: quit   C: clear   SPACE: add space",
            (20, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)

        cv2.imshow("Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence.clear()
            print("Sentence cleared.")
        elif key == ord(' '):
            sentence.append(' ')

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal sentence: {''.join(sentence).strip()}")