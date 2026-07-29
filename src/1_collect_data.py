import cv2
import csv
import os
import sys
import pandas as pd
import mediapipe as mp

sys.path.append(os.path.dirname(__file__))
from utils import extract_landmarks, get_hands

GESTURES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'Hello','Yes','No','Thanks'
]
SAMPLES_PER_GESTURE = 200
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures.csv')

os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

# Check already collected gestures
already_done = {}
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    counts = df['label'].value_counts()
    for gesture in GESTURES:
        if gesture in counts and counts[gesture] >= SAMPLES_PER_GESTURE:
            already_done[gesture] = counts[gesture]
else:
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'{axis}{i}' for i in range(21) for axis in ['x','y','z']] + ['label']
        writer.writerow(header)

# Filter remaining gestures
remaining = [g for g in GESTURES if g not in already_done]

print("=== DATA COLLECTION ===")
print(f"Already done  : {list(already_done.keys())}")
print(f"Remaining     : {remaining}")
print(f"Samples each  : {SAMPLES_PER_GESTURE}")

if not remaining:
    print("\n✅ All gestures already collected! Run 2_train_model.py")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_draw  = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with get_hands() as hands:
    for gesture in remaining:
        count = 0
        print(f"\n>>> Gesture: '{gesture}' — press SPACE when ready")

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (20, 20, 20), -1)
            cv2.putText(frame, f"Next: '{gesture}' — Press SPACE when ready",
                        (20, 48), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)
            # Show progress in corner
            progress_text = f"Done: {len(already_done)+remaining.index(gesture)}/{len(GESTURES)}"
            cv2.putText(frame, progress_text,
                        (frame.shape[1]-280, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 1)
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        while count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            coords, hand_lms = extract_landmarks(frame, hands)

            if coords is not None:
                with open(DATA_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(list(coords) + [gesture])
                count += 1
                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 120), thickness=2),
                    mp_draw.DrawingSpec(color=(255, 0, 200), thickness=2)
                )

            bar_w = int((count / SAMPLES_PER_GESTURE) * 600)
            cv2.rectangle(frame, (40, frame.shape[0]-50), (640, frame.shape[0]-20), (50,50,50), -1)
            cv2.rectangle(frame, (40, frame.shape[0]-50), (40+bar_w, frame.shape[0]-20), (0,200,100), -1)
            cv2.putText(frame, f"'{gesture}'  {count}/{SAMPLES_PER_GESTURE}",
                        (20, 48), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 100), 2)
            cv2.imshow("Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\nQuit at '{gesture}' — progress saved!")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        already_done[gesture] = SAMPLES_PER_GESTURE
        print(f"  ✅ '{gesture}' done")

cap.release()
cv2.destroyAllWindows()
print("\n🎉 All gestures collected! Now run: python 2_train_model.py")