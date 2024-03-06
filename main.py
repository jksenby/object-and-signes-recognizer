import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_draw = mp.solutions.drawing_utils

model = load_model("gesture_model_v2.h5")

def detect_hand(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame, results

def preprocess_image(img):
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.stack((img,) * 3, axis=-1)
    img = img.astype('float32') / 255.0
    return img

def recognize_gesture():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_with_hand, results = detect_hand(frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            processed_img = preprocess_image(frame)
            processed_img = np.expand_dims(processed_img, axis=0)

            prediction = model.predict(processed_img)
            predicted_class = np.argmax(prediction)
            predicted_letter = chr(65 + predicted_class)

            cv2.putText(frame_with_hand, predicted_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Gesture Recognition', frame_with_hand)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

recognize_gesture()
