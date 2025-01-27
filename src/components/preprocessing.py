import os
import mediapipe as mp
import cv2
import pickle

data_path = '/content/dataset/asl_alphabet_train'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,  # only one hand
    min_detection_confidence=0.3
)

data = []
labels = []

for dir_name in os.listdir(data_path):
    dir_path = os.path.join(data_path, dir_name)
    if not os.path.isdir(dir_path):
        continue 

    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        
        if not os.path.isfile(img_path):
            continue  

        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue  

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # only first hand, for even dataset
            hand_landmarks = results.multi_hand_landmarks[0]

            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]

            # normalize values
            min_x = min(x_coords)
            min_y = min(y_coords)
            normalized_landmarks = []
            for x, y in zip(x_coords, y_coords):
                normalized_landmarks.append(x - min_x)
                normalized_landmarks.append(y - min_y)

            # normalize
            data.append(normalized_landmarks)
            labels.append(dir_name)
        else:
            continue  

hands.close()

import numpy as np

data = np.array(data)
labels = np.array(labels)

# save
with open('data.pickle', 'wb') as file:
    pickle.dump({'data': data, 'labels': labels}, file)

print("Preprocessing completed successfully.")
print(f"Total samples collected: {len(data)}")