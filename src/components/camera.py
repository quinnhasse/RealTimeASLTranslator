import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model
model_dict = pickle.load(open('/Users/quinnhasse/RealTimeASLTRasnlator/model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)  # camera index changed

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Labels
labels_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'nothing',
    15: 'O',
    16: 'P',
    17: 'Q',
    18: 'R',
    19: 'S',
    20: 'space',
    21: 'T',
    22: 'U',
    23: 'V',
    24: 'W',
    25: 'X',
    26: 'Y',
    27: 'Z'
}

reverse_labels_dict = {v: k for k, v in labels_dict.items()}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Couldn't get frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            # check prediction string issue
            if isinstance(prediction[0], str):
                predicted_character = prediction[0]
            else:
                predicted_character = labels_dict.get(int(prediction[0]), "unknown")

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # frame
    cv2.imshow('frame', frame)

    # exit by hitting escape
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()