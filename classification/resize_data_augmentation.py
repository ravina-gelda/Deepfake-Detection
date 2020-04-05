import os
import argparse
from os.path import join
import cv2
import dlib
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CUDA_VISIBLE_DEVICES = 0
input_shape = (64, 64, 3)
train_dir = '../processed_image/'
data = [f for f in os.listdir(train_dir) if f.endswith('.png')]
X = []
whole_label = pd.read_csv('idx_label.csv')
y_true_label = whole_label['File_Label']
y_generated_label = whole_label['Machine_Label']
a1 = dict(Counter(y_true_label))
a2 = dict(Counter(y_generated_label))
print(a1)
print(a2)
for idx in range(1, 35476):
    im = Image.open('../processed_image/'+'_'+str(idx)+'.png')
    im = im.resize((64, 64))
    im = np.array(im)
    cv2.imwrite('../augmented_image_file_label/'+'_'+str(idx-1)+'.png', im)
    cv2.imwrite('../augmented_image_machine_label/'+'_'+str(idx-1)+'.png', im)
new_idx_true = 35474
new_idx_machine = 35474
for idx in range(0, 35475):
    if y_true_label[idx] == 1:
        im = Image.open('../augmented_image_file_label/'+'_'+str(idx)+'.png')
        hoz_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
        hoz_flip = np.array(hoz_flip)
        new_idx_true += 1
        cv2.imwrite('../augmented_image_file_label/'+'_'+str(new_idx_true)+'.png', hoz_flip)
        ver_flip = im.transpose(Image.FLIP_TOP_BOTTOM)
        ver_flip = np.array(ver_flip)
        new_idx_true += 1
        cv2.imwrite('../augmented_image_file_label/'+'_'+str(new_idx_true)+'.png', ver_flip)
        rot_fli = im.rotate(15)
        rot_fli = np.array(rot_fli)
        new_idx_true += 1
        cv2.imwrite('../augmented_image_file_label/'+'_'+str(new_idx_true)+'.png', rot_fli)
    if y_generated_label[idx] == 1:
        im = Image.open('../augmented_image_machine_label/'+'_'+str(idx)+'.png')
        hoz_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
        hoz_flip = np.array(hoz_flip)
        new_idx_machine += 1
        cv2.imwrite('../augmented_image_machine_label/'+'_'+str(new_idx_machine)+'.png', hoz_flip)
        ver_flip = im.transpose(Image.FLIP_TOP_BOTTOM)
        ver_flip = np.array(ver_flip)
        new_idx_machine += 1
        cv2.imwrite('../augmented_image_machine_label/'+'_'+str(new_idx_machine)+'.png', ver_flip)
        







