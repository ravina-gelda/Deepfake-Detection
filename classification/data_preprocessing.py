import sys
import csv
import getopt
import pickle
import os
import argparse
from os.path import join
import cv2
import glob
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import json
import detect_from_video
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
from detect_from_video import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path = '../classification/pretrainedmodels/faceforensics++_models_subset/full/xception/full_raw.p'
model = torch.load(model_path)
model = model.cuda()
train_dir = '../dataset/train_sample_videos/'
all_train_videos = glob.glob(os.path.join(train_dir, '*.mp4'))
metadata = pd.read_json('../dataset/train_sample_videos/metadata.json').T
print(len(metadata))
print(len(metadata[metadata.label == "REAL"]),len(metadata[metadata.label == "FAKE"]))
metadata.index.name = 'filename'
print(metadata.head())
data = metadata['label']
print(data['aagfhgtpmv.mp4'])
df = pd.DataFrame(columns=['filename', 'distance', 'label'])
df.head()
df.to_csv('train.csv', index=False)
with open(os.path.join(train_dir, 'metadata.json'), 'r') as file:
    data = json.load(file)
list_of_train_data = [f for f in os.listdir(train_dir) if f.endswith('.mp4')]
def generate_image_and_label(train_dir, list_of_train_data, model, start_frame=0, end_frame=None, cuda=True):
    idx = 0
    label = []
    label1 = []
    n_video = 0
    print(len(list_of_train_data))
    for v_id in list_of_train_data:
        print(n_video)
        n_video += 1
        print(v_id)
        face_detector = dlib.get_frontal_face_detector()
        reader = cv2.VideoCapture(os.path.join(train_dir, v_id))
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
                face = faces[0]
                x, y, size = get_boundingbox(face, width, height)
                cropped_face = image[y:y+size, x:x+size]
                prediction, output = predict_with_model(cropped_face, model,
                                                    cuda=cuda)
                if data[v_id]['label'] == 'REAL':
                    prediction2 = 1
                else:
                    prediction2 = 0
                label.append(prediction)
                label1.append(prediction2)
                idx += 1
                cv2.imwrite('../processed_image/'+'_'+str(idx)+'.png', cv2.resize(cropped_face, (128, 128)))
            if frame_num >= end_frame:
                break
    print(idx)
    print(label)
    print(len(label))
    print(label1)
    print(len(label1))
    writer1 = csv.writer(open('idx_label.csv','w'))
    writer1.writerow(['Index', 'Machine_Label','File_Label'])
    for i in range(idx):
        writer1.writerow([i, label[i], label1[i]])
    
train_dir = '../dataset/train_sample_videos/'
generate_image_and_label(train_dir, list_of_train_data, model = model, start_frame=0, end_frame=100, cuda=True)
            
            

