import cv2
import numpy as np
import mediapipe as mp

def get_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

def extract_landmarks(frame, hands_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_model.process(rgb)
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        coords = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
        return coords, result.multi_hand_landmarks[0]
    return None, None