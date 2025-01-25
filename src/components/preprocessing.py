import os
import mediapipe as mp
import cv2

data_path = '/Users/quinnhasse/RealTimeASLTRasnlator/src/notebook/data/asl_alphabet_test'

for dir_ in os.listdir(data_path):
    for img_path in os.listdir(os.path.join(data_path,dir_)):
        img = cv2.imread(os.path.join(data_path,dir_,img_path))