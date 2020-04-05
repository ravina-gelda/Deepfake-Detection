
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
import seaborn as sns
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
whole_label = pd.read_csv('idx_label.csv')
y_true_label = whole_label['File_Label']
y_generated_label = whole_label['Machine_Label']
a1 = dict(Counter(y_true_label))
a2 = dict(Counter(y_generated_label))
print(a1)
print(a2)
metadata = pd.read_json('../dataset/train_sample_videos/metadata.json').T
print(len(metadata))
print(len(metadata[metadata.label == "REAL"]),len(metadata[metadata.label == "FAKE"]))
