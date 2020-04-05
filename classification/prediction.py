import os
import argparse
from os.path import join
import cv2
import dlib
import csv
import json
import seaborn as sns
from PIL import Image as pil_image
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import detect_from_video
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
from tensorflow.keras.models import load_model
test_dir = '../dataset/test_new_videos/'
whole_label = pd.read_csv('idx_label.csv')
y_true_label = whole_label['File_Label']
y_generated_label = whole_label['Machine_Label']
a2 = dict(Counter(y_generated_label))
print(a2)
metadata = pd.read_json('../dataset/test_new_videos/metadata0.json').T
print(len(metadata))
print(len(metadata[metadata.label == "REAL"]),len(metadata[metadata.label == "FAKE"]))
prop1 = a2[1]/len(metadata[metadata.label == "REAL"])
prop2 = a2[0]/len(metadata[metadata.label == "FAKE"])
threshold = prop1/(prop1 + prop2)
print(threshold)
list_of_test_data = [f for f in os.listdir(test_dir) if f.endswith('.mp4')]
print(len(list_of_test_data))
def final_evaluation(test_dir, list_of_test_data, model, threshold, start_frame=0, end_frame=None, cuda=False):
    w_real = 1 - threshold
    w_fake = threshold
    pred = np.zeros([len(list_of_test_data), 2])
    label = []
    label1 = []
    n_video = 0
    print(len(list_of_test_data))
    for v_id in list_of_test_data:
        count_real = 0
        count_fake = 0
        n_video += 1
        print(v_id)
        print(n_video)
        face_detector = dlib.get_frontal_face_detector()
        reader = cv2.VideoCapture(os.path.join(test_dir, v_id))
        num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        print(num_frames)
        frame_num = 0
        end_frame = end_frame if end_frame else num_frames
        pbar = tqdm(total=end_frame-start_frame)
        while reader.isOpened():
            _, image = reader.read()
            if image is None:
                break
            frame_num += 1
            if frame_num < start_frame:
                continue
            pbar.update(1)
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 1)
            if len(faces):
                # For now only take biggest face
                face = faces[0]

                # --- Prediction ---------------------------------------------------
                # Face crop with dlib and bounding box scale enlargement
                x, y, size = detect_from_video.get_boundingbox(face, width, height)
                cropped_face = image[y:y+size, x:x+size]
                input = np.array(img_to_array(cv2.resize(cropped_face, (64, 64))))
                input = input.reshape(-1, 64, 64, 3)
                prediction = model.predict(input)
#                print(frame_num)
#                print(prediction)
#                print(prediction[0][0])
#                print(prediction[0][1])
                if prediction[0][0] == 1.0:
#                    print(count_fake)
                    count_fake += 1
                    pred[n_video-1][0] += w_fake
#                    print(pred[n_video-1][0])
                elif prediction[0][1] == 1.0:
#                    print(count_real)
                    count_real += 1
                    pred[n_video-1][1] += w_real
#                    print(pred[n_video-1][1])
            
            
            if frame_num >= end_frame:
                break
    return pred

def submit(pred, path):
    for i in range(pred.shape[0]):
        if pred[i][0] == 0 and pred[i][1] == 0:
            pred[i][0] = 0.5
            pred[i][1] = 0.5
        else:
            pred[i][0] /= (pred[i][0] + pred[i][1])
            pred[i][1] /= (pred[i][0] + pred[i][1])
    writer = csv.writer(open(path, 'w'))
    writer.writerow(['filename', 'label'])
    n = 0
    for v_id in list_of_test_data:
        if pred[i][0] > pred[i][1]:
            writer.writerow([v_id, 0])
            n += 1
        else:
            writer.writerow([v_id, 1])
            n += 1
            
model = load_model('../classification/trained_models/basic_cnn_train_with_true_label.h5')
pred = final_evaluation(test_dir, list_of_test_data, model, threshold, start_frame=0, end_frame=25, cuda=True)
path = '../submision/basic_cnn_train_with_true_label.csv'
submit(pred, path)
#
#model = load_model('basic_cnn_train_with_generated_label.h5')
#pred = final_evaluation(test_dir, list_of_test_data, model, threshold, start_frame=0, end_frame=25, cuda=True)
#path = '../submision/basic_cnn_train_with_generated_label.csv'
#submit(pred, path)
#
#model = load_model('basic_cnn_train_with_true_label_imbalance.h5')
#pred = final_evaluation(test_dir, list_of_test_data, model, threshold, start_frame=0, end_frame=25, cuda=True)
#path = '../submision/basic_cnn_train_with_true_label_imbalance.csv'
#submit(pred, path)



            
            

